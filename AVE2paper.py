"""
@author: LXA
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import DNN_base
import AVE_infos
import AVE_tools
import AVE_LogPrint


#  [a,b] 生成随机数, n代表变量个数
def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(variable_dim, batch_size) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


def solve_AVE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    AVE_LogPrint.dictionary_out2file(R, log_fileout)

    batch_size = R['batch_size']
    penalty2WB = R['penalty2weight_biases']
    penalty2func_init = R['penalty_func']
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']

    activate_func = R['activate_func']

    row = R['Row']                             # 自变量的维数（列向量）
    column = R['Column']
    dim2train_test = R['dim2train_test']

    if R['problem_type'] == 'square_matrix':
        Mat_A, Mat_B, u_true, f_side = AVE_infos.get_AVE_infos(Row=row, Column=column, AVE_equa_name=R['eqs_name'])
        dim2problem = column
        if R['eqs_name'] == 'matrix2' or R['eqs_name'] == 'matrix7':
            dim2problem = column * column
    elif R['problem_type'] == 'no_square_matrix':
        Mat_A, Mat_B, u_true, f_side = AVE_infos.get_AVE_infos2no_square(Row=row, Column=column, AVE_equa_name=R['eqs_name'])
        dim2problem = column

    flag = 'WB'
    hidden_layers = R['hidden_layers']
    if R['model'] != 'AVE_DNN_Fourier':
        Weights, Biases = DNN_base.Xavier_init_NN(dim2train_test, dim2problem, hidden_layers, flag)
    else:
        Weights, Biases = DNN_base.Xavier_init_NN_Fourier(dim2train_test, dim2problem, hidden_layers, flag)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[dim2train_test, batch_size])
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            func_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='func_p')
            if R['model'] == 'AVE_DNN':
                UNN = DNN_base.DNN(X_it, Weights, Biases, hidden_layers, activateIn_name=R['activate_func2in'],
                                   activate_name=activate_func, activateOut_name=R['activate_func2out'])
            elif R['model'] == 'AVE_DNN_Scale':
                freq = R['freq']
                UNN = DNN_base.DNN_scale(X_it, Weights, Biases, freq, hidden_layers,
                                         activateIn_name=R['activate_func2in'], activate_name=activate_func,
                                         activateOut_name=R['activate_func2out'])
            elif R['model'] == 'AVE_DNN_Fourier':
                freq = R['freq']
                UNN = DNN_base.DNN_scale(X_it, Weights, Biases, freq, hidden_layers, activate_name=activate_func,
                                         activateOut_name=R['activate_func2out'])

            A_U = tf.matmul(Mat_A, UNN)
            B_absU = tf.matmul(Mat_B, tf.abs(UNN))
            square_error = tf.square(A_U - B_absU - f_side)
            norm_square2error = tf.reshape(tf.reduce_sum(square_error, axis=0), shape=[-1, 1])
            if R['sqrt_error'] == 1:
                loss_AVE = tf.reduce_mean(tf.sqrt(norm_square2error))
                norm2_right_hand = tf.sqrt(tf.reduce_sum(tf.square(f_side)))
                func_res = loss_AVE / norm2_right_hand
            else:
                loss_AVE = tf.reduce_mean(norm_square2error)
                norm2_right_hand = tf.reduce_sum(tf.square(f_side))
                func_res = loss_AVE / norm2_right_hand

            if R['regular_weight_model'] == 'L1':
                regular_WB = DNN_base.regular_weights_biases_L1(Weights, Biases)  # 正则化权重参数 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = DNN_base.regular_weights_biases_L2(Weights, Biases)  # 正则化权重参数 L2正则化
            else:
                regular_WB = 0.0

            PWB = penalty2WB * regular_WB
            loss = func_penalty*loss_AVE + PWB                        # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            loss_optimizer = my_optimizer.minimize(loss, global_step=global_steps)

            component_wise_error = tf.reshape(tf.reduce_mean(tf.square(u_true - UNN), axis=-1), shape=[-1, 1])
            square_solution_error = tf.reshape(tf.reduce_sum(tf.square(u_true - UNN), axis=0), shape=[-1, 1])
            if R['sqrt_error'] == 1:
                solution_error = tf.sqrt(tf.reduce_mean(square_solution_error))
                solution_residual = solution_error / tf.sqrt(tf.reduce_sum(tf.square(u_true)))
            else:
                solution_error = tf.reduce_mean(square_solution_error)
                solution_residual = solution_error / tf.reduce_sum(tf.square(u_true))

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    loss_all, solu_mse_all, solu_rel_all, function_error_all, function_residual_all = [], [], [], [], []
    epoch_1000 = []

    # 生成训练集时的区间选择
    region_l = 0.0
    region_r = 1.0
    # 生成数据，用于测试训练后的网络
    test_bach_size = 1
    R['test_x'] = rand_it(test_bach_size, dim2train_test, region_l, region_r)

    # ConfigProto 加上 allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = rand_it(batch_size, dim2train_test, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['stage_penalty_func'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_func = penalty2func_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_func = 10 * penalty2func_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_func = 50 * penalty2func_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_func = 100 * penalty2func_init
                elif i_epoch < int(4 * R['max_epoch'] / 5):
                    temp_penalty_func = 500 * penalty2func_init
                else:
                    temp_penalty_func = 1000 * penalty2func_init
                    # temp_penalty_func = 500 * penalty2func_init
            else:
                temp_penalty_func = penalty2func_init

            _, loss_tmp, comp_err, sol_error_tmp, sol_rel_tmp, fun_error_temp, func_res_temp, pwb = sess.run(
                [loss_optimizer, loss, component_wise_error, solution_error, solution_residual, loss_AVE,
                 func_res, PWB], feed_dict={X_it: x_it_batch, in_learning_rate: tmp_lr,
                                            func_penalty: temp_penalty_func})

            if i_epoch % 1000 == 0:
                run_time_temp = time.time() - t0
                AVE_LogPrint.log_Print2one_epoch(i_epoch, run_time_temp, tmp_lr, temp_penalty_func, pwb, sol_error_tmp,
                                                 sol_rel_tmp, loss, fun_error_temp, func_res_temp, log_out=log_fileout)
                epoch_1000.append(i_epoch / 1000)
                loss_all.append(loss_tmp)
                solu_mse_all.append(sol_error_tmp)
                solu_rel_all.append(sol_rel_tmp)
                function_error_all.append(fun_error_temp)
                function_residual_all.append(func_res_temp)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'AVE'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 120000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ---------------------------- Setup of PDE_DNNs -------------------------------
    R['problem_type'] = 'square_matrix'
    # R['problem_type'] = 'no_square_matrix'

    if R['problem_type'] == 'square_matrix':
        # R['eqs_name'] = 'matrix1'
        # R['eqs_name'] = 'matrix2'
        # R['eqs_name'] = 'matrix3'
        # R['eqs_name'] = 'matrix4'
        # R['eqs_name'] = 'matrix5'
        # R['eqs_name'] = 'matrix6'
        R['eqs_name'] = 'matrix7'
        # R['eqs_name'] = 'Hilbert'

        R['Row'] = 200
        R['Column'] = 200
        # R['Row'] = 500
        # R['Column'] = 500
        # R['Row'] = 1000
        # R['Column'] = 1000
        if R['eqs_name'] == 'matrix2' or R['eqs_name'] == 'matrix7':
            # R['Row'] = 10
            # R['Column'] = 10
            # R['Row'] = 20
            # R['Column'] = 20
            R['Row'] = 30
            R['Column'] = 30
    elif R['problem_type'] == 'no_square_matrix':
        R['eqs_name'] = 'matrix0'
        # R['eqs_name'] = 'matrix1'

        R['Row'] = 50
        R['Column'] = 50

    # R['dim2train_test'] = 50        # 输入维数，即训练数据的维度大小
    # R['dim2train_test'] = 100       # 输入维数，即训练数据的维度大小
    # R['dim2train_test'] = 150       # 输入维数，即训练数据的维度大小
    R['dim2train_test'] = 200         # 输入维数，即训练数据的维度大小

    if R['dim2train_test'] <= 500:
        # R['batch_size'] = 200       # 内部训练数据的批大小
        # R['batch_size'] = 250       # 内部训练数据的批大小
        R['batch_size'] = 500         # 内部训练数据的批大小
    else:
        R['batch_size'] = 750

    R['lr_decay'] = 5e-5                            # 学习率 decay
    R['learning_rate'] = 2e-4                       # 学习率
    R['optimizer_name'] = 'Adam'                    # 优化器

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'

    R['sqrt_error'] = 0
    if R['sqrt_error'] == 0:
        R['penalty2weight_biases'] = 0.00001  # Regularization parameter for weights

        R['stage_penalty_func'] = 1
        if R['stage_penalty_func'] == 0:
            R['penalty_func'] = 20000
        else:
            R['penalty_func'] = 1000
    else:
        R['penalty2weight_biases'] = 0.001  # Regularization parameter for weights

        R['stage_penalty_func'] = 0
        if R['stage_penalty_func'] == 0:
            R['penalty_func'] = 1000
        else:
            R['penalty_func'] = 100

    if R['dim2train_test'] <= 300:
        # R['hidden_layers'] = (150, 100, 80, 60, 40)
        # R['hidden_layers'] = (60, 50, 40, 30, 20)
        # R['hidden_layers'] = (150, 100, 80, 60, 60, 40)
        R['hidden_layers'] = (200, 150, 100, 60, 60, 40)
        # R['hidden_layers'] = (200, 150, 100, 80, 60, 40)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 100, 50)
    else:
        # R['hidden_layers'] = (200, 150, 100, 50)
        R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100, 50)

    # R['model'] = 'AVE_DNN'                         # 使用的网络模型
    # R['model'] = 'AVE_DNN_BN'
    # R['model'] = 'AVE_DNN_scale'
    R['model'] = 'AVE_DNN_Fourier'

    # 网络的频率范围设置
    # R['freq'] = np.arange(1, 31)
    # R['freq'] = np.arange(1, 51)
    R['freq'] = np.arange(1, 101)
    # R['freq'] = np.random.normal(1, 120, 100)

    # R['activate_func2in'] = 'relu'
    # R['activate_func2in'] = 'leakly_relu'           # 这个激活函数有问题, 会不收敛
    # R['activate_func2in'] = 'elu'
    # R['activate_func2in'] = 'gelu'
    # R['activate_func2in'] = 'tan'           # 这个激活函数有问题, 会不收敛
    # R['activate_func2in'] = 'sin'
    # R['activate_func2in'] = 'srelu'
    R['activate_func2in'] = 's2relu'
    # R['activate_func2in'] = 'phi'

    # R['activate_func'] = 'relu'
    # R['activate_func'] = 'leakly_relu'           # 这个激活函数有问题, 会不收敛
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'gelu'
    # R['activate_func'] = 'tanh'           # 这个激活函数有问题, 会不收敛
    # R['activate_func'] = 'sin'
    # R['activate_func'] = 'srelu'
    R['activate_func'] = 's2relu'
    # R['activate_func'] = 'phi'

    R['activate_func2out'] = 'linear'

    solve_AVE(R)