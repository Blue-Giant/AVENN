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


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    AVE_tools.log_string('The type of problem: %s\n' % str(R_dic['problem_type']), log_fileout)
    AVE_tools.log_string('The name of equation: %s\n' % str(R_dic['eqs_name']), log_fileout)
    AVE_tools.log_string('Model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    AVE_tools.log_string('Activate function for solving problem: %s\n' % str(R_dic['activate_func']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        AVE_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        AVE_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    AVE_tools.log_string('The initial learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    AVE_tools.log_string('The hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    AVE_tools.log_string('The model of regularizing weights: %s\n' % str(R_dic['regular_weight_model']), log_fileout)
    if R_dic['sqrt_error'] == 1:
        AVE_tools.log_string('The model of solution error and function error: %s\n' % str('sqrt_error'), log_fileout)
    else:
        AVE_tools.log_string('The model of solution error and function error: %s\n' % str('square_error'), log_fileout)

    if R_dic['activate_stop'] != 0:
        AVE_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        AVE_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


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
    dictionary_out2file(R, log_fileout)

    batch_size = R['batch_size']
    beta = R['regular_weight']          # Regularization parameter for weights
    penalty2func_init = R['penalty_func']
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']

    activate_func = R['activate_func']

    row = R['Row']          # 自变量的维数（列向量）
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
    Weights, Biases = DNN_base.initialize_NN_xavier_right(dim2train_test, dim2problem, hidden_layers, flag)
    # Weights, Biases = AVE_DNN_base2.initialize_NN_random_normal(dim2train_test, dim2problem, hidden_layers, flag)
    # Weights, Biases = AVE_DNN_base2.initialize_NN_random_normal2(dim2train_test, dim2problem, hidden_layers, flag)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[dim2train_test, None])
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            func_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='func_p')
            if R['model'] == 'AVE_DNN':
                U_hat = DNN_base.DNN(X_it, Weights, Biases, hidden_layers, AVE_act_name=activate_func)
            elif R['model'] == 'AVE_DNN_BN':
                U_hat = DNN_base.AVE_DNN_BN(X_it, Weights, Biases, hidden_layers, AVE_act_name=activate_func)
            else:
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_hat = DNN_base.AVE_DNN_scale(X_it, Weights, Biases, freq, hidden_layers, AVE_act_name=activate_func)

            A_U = tf.matmul(Mat_A, U_hat)
            B_absU = tf.matmul(Mat_B, tf.abs(U_hat))
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
                regular_WB = AVE_DNN_base2.regular_weights_biases_L1(Weights, Biases)  # 正则化权重参数 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = AVE_DNN_base2.regular_weights_biases_L2(Weights, Biases)  # 正则化权重参数 L2正则化
            else:
                regular_WB = 0.0

            penalty_WB = beta * regular_WB
            loss = func_penalty*loss_AVE + beta * regular_WB                        # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss_optimizer = my_optimizer.minimize(loss, global_step=global_steps)

            square_solution_error = tf.reshape(tf.reduce_sum(tf.square(u_true - U_hat), axis=0), shape=[-1, 1])
            if R['sqrt_error'] == 1:
                solution_error = tf.sqrt(tf.reduce_mean(square_solution_error))
                solution_residual = solution_error / tf.sqrt(tf.reduce_sum(tf.square(u_true)))
            else:
                solution_error = tf.reduce_mean(square_solution_error)
                solution_residual = solution_error / tf.reduce_sum(tf.square(u_true))

    t0 = time.time()
    # 空列表, 使用 append() 添加元素
    loss_all, solution_error_all, solution_residual_all, function_error_all, function_residual_all = [], [], [], [], []
    testing_solution_error = []
    testing_solution_residual = []
    testing_function_error = []
    testing_function_residual = []
    testing_epoch = []

    # 问题区域
    region_l = 0.0
    region_r = 1.0
    # 生成数据，用于测试训练后的网络
    test_bach_size = 1
    R['test_x'] = rand_it(test_bach_size, dim2train_test, region_l, region_r)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
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
                # elif i_epoch < int(3 * R['max_epoch'] / 4):
                #     temp_penalty_func = 200 * penalty2func_init
                else:
                    temp_penalty_func = 200 * penalty2func_init
                    # temp_penalty_func = 500 * penalty2func_init
            else:
                temp_penalty_func = penalty2func_init

            _, loss_tmp, sol_error_tmp, sol_res_tmp, fun_error_temp, func_res_temp, pwb = sess.run(
                [train_my_loss_optimizer, loss, solution_error, solution_residual, loss_AVE, func_res, penalty_WB],
                feed_dict={X_it: x_it_batch, in_learning_rate: tmp_lr, func_penalty: temp_penalty_func})

            loss_all.append(loss_tmp)
            solution_error_all.append(sol_error_tmp)
            solution_residual_all.append(sol_res_tmp)
            function_error_all.append(fun_error_temp)
            function_residual_all.append(func_res_temp)

            if i_epoch % 1000 == 0:
                # 将运行结果打印出来
                print('train epoch: %d, time: %.5f' % (i_epoch, time.time() - t0))
                print('learning rate: %.10f' % tmp_lr)
                print('penalty of function error: %.10f' % temp_penalty_func)
                print('weights and biases with penalty: %.10f' % pwb)
                print('total loss for training: %.10f' % loss_tmp)
                print('solution error for training: %.10f' % sol_error_tmp)
                print('solution residual for training: %.12f' % sol_res_tmp)
                print('function error for training: %.10f' % fun_error_temp)
                print('function residual error for training: %.12f\n' % func_res_temp)

                AVE_tools.log_string('train epoch: %d,time: %.5f' % (i_epoch, time.time() - t0), log_fileout)
                AVE_tools.log_string('learning rate: %.10f' % tmp_lr, log_fileout)
                AVE_tools.log_string('penalty of function error: %.10f' % temp_penalty_func, log_fileout)
                AVE_tools.log_string('weights and biases with penalty: %.10f' % pwb, log_fileout)
                AVE_tools.log_string('total loss for training: %.10f' % loss_tmp, log_fileout)
                AVE_tools.log_string('solution error for training: %.10f' % sol_error_tmp, log_fileout)
                AVE_tools.log_string('solution residual for training: %.12f' % sol_res_tmp, log_fileout)
                AVE_tools.log_string('function error for training: %.10f' % fun_error_temp, log_fileout)
                AVE_tools.log_string('function residual error for training: %.12f\n' % func_res_temp, log_fileout)

                # -----------------------  save data to a array, then plot them ---------------------------------
                R['training loss'] = np.array(loss_all)
                R['solution error'] = np.array(solution_error_all)
                R['solution residual'] = np.array(solution_residual_all)
                R['function error'] = np.array(function_error_all)
                R['function residual'] = np.array(function_residual_all)

                # ---------------------------   test network ----------------------------------------------
                testing_epoch.append(i_epoch / 1000)
                U_predict, func_err2test, func_res2test = sess.run([U_hat, loss_AVE, func_res],
                                                                   feed_dict={X_it: R['test_x']})

                if R['sqrt_error'] == 1:
                    solu_err2test = np.sqrt(np.sum(np.square(u_true - U_predict)))
                    solu_res2test = solu_err2test / np.sqrt(np.sum(np.square(u_true)))
                else:
                    solu_err2test = np.sum(np.square(u_true - U_predict))
                    solu_res2test = solu_err2test / np.sum(np.square(u_true))

                testing_solution_error.append(solu_err2test)
                testing_solution_residual.append(solu_res2test)
                testing_function_error.append(func_err2test)
                testing_function_residual.append(func_res2test)

                print('Norm2 error of solution_diff for testing: %.10f' % solu_err2test)
                print('Residual error of solution_diff for testing: %.12f' % solu_res2test)
                print('Norm2 error of function for testing: %.10f' % func_err2test)
                print('Residual error of function for testing: %.12f\n\n' % func_res2test)

                AVE_tools.log_string('Norm2 error of solution_diff for testing: %.10f' % solu_err2test, log_fileout)
                AVE_tools.log_string('Residual of solution_diff for testing: %.12f' % solu_res2test, log_fileout)
                AVE_tools.log_string('Norm2 error of function for testing: %.10f' % func_err2test, log_fileout)
                AVE_tools.log_string('Residual error of function for testing: %.12f\n\n' % func_res2test, log_fileout)

        # -----------------------  plot the training results into figures ---------------------------------
        plt.figure()
        ax = plt.gca()
        plt.plot(R['training loss'], 'g-.', label='loss')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.legend(fontsize=18)
        plt.title('loss', fontsize=15)
        fntmp = '%s/%strain_loss' % (R['FolderName'], R['seed'])
        AVE_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)

        plt.figure()
        ax = plt.gca()
        plt.plot(R['solution error'], 'r-.', label='solu_err')
        plt.plot(R['solution residual'], 'b:', label='solu_res')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('error', fontsize=18)
        plt.legend(fontsize=18)
        plt.title('solution error', fontsize=15)
        fntmp = '%s/%ssolu_error2train' % (R['FolderName'], R['seed'])
        AVE_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)

        plt.figure()
        ax = plt.gca()
        plt.plot(R['function error'], 'r-.', label='func_err')
        plt.plot(R['function residual'], 'b:', label='func_res')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('error', fontsize=18)
        plt.legend(fontsize=18)
        plt.title('function error', fontsize=15)
        fntmp = '%s/%sfunc_error2train' % (R['FolderName'], R['seed'])
        AVE_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)

        # ----------------------  plot the testing results into figures --------------------------------
        plt.figure()
        ax = plt.gca()
        plt.plot(testing_epoch, testing_solution_error, 'r-.', label='solu_err')
        plt.plot(testing_epoch, testing_solution_residual, 'b:', label='solu_res')
        plt.xlabel('epoch/1000', fontsize=18)
        plt.ylabel('error', fontsize=18)
        ax.set_yscale('log')
        plt.legend(fontsize=18)
        plt.title('solution error', fontsize=15)
        fntmp = '%s/%ssolu_error2test' % (R['FolderName'], R['seed'])
        AVE_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)

        plt.figure()
        ax = plt.gca()
        plt.plot(testing_epoch, testing_function_error, 'r-.', label='func_err')
        plt.plot(testing_epoch, testing_function_residual, 'b:', label='func_res')
        plt.xlabel('epoch/1000', fontsize=18)
        plt.ylabel('error', fontsize=18)
        ax.set_yscale('log')
        plt.legend(fontsize=18)
        plt.title('function error', fontsize=15)
        fntmp = '%s/%sfunc_error2test' % (R['FolderName'], R['seed'])
        AVE_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)

        # print('predict solution of last epoch:', U_predict)
        fntmp1 = '%s/%su_predict.txt' % (R['FolderName'], R['seed'])
        np.savetxt(fntmp1, U_predict)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 1
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'pos1'
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
        # R['eqs_name'] = 'matrix7'
        # R['eqs_name'] = 'Hilbert'

        R['Row'] = 200
        R['Column'] = 200
    elif R['problem_type'] == 'no_square_matrix':
        R['eqs_name'] = 'matrix0'
        # R['eqs_name'] = 'matrix1'

        R['Row'] = 50
        R['Column'] = 50

    # R['dim2train_test'] = 50        # 输入维数，即训练数据的维度大小
    R['dim2train_test'] = 100         # 输入维数，即训练数据的维度大小
    # R['dim2train_test'] = 150       # 输入维数，即训练数据的维度大小
    # R['dim2train_test'] = 200       # 输入维数，即训练数据的维度大小

    if R['dim2train_test'] <= 500:
        R['batch_size'] = 200         # 内部训练数据的批大小
        # R['batch_size'] = 250       # 内部训练数据的批大小
        # R['batch_size'] = 400       # 内部训练数据的批大小
    else:
        R['batch_size'] = 500

    R['lr_decay'] = 5e-5                            # 学习率 decay
    R['learning_rate'] = 2e-4                       # 学习率
    R['optimizer_name'] = 'Adam'                    # 优化器
    R['sqrt_error'] = 1
    R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight'] = 0.005                     # Regularization parameter for weights

    # R['stage_penalty_func'] = 0
    # R['penalty_func'] = 1000

    R['stage_penalty_func'] = 1
    R['penalty_func'] = 50

    if R['dim2train_test'] <= 300:
        # R['hidden_layers'] = (60, 50, 40, 30, 20)
        # R['hidden_layers'] = (150, 100, 100, 80, 60, 60, 40)
        R['hidden_layers'] = (150, 100, 80, 80, 60, 40, 40)
    else:
        # R['hidden_layers'] = (200, 150, 100, 50)
        R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100, 50)

    # R['model'] = 'AVE_DNN'                # 使用的网络模型
    R['model'] = 'AVE_DNN_BN'             # 使用的网络模型
    # R['model'] = 'AVE_DNN_scale'

    R['activate_func'] = 'srelu'
    # R['activate_func'] = 'relu'
    # R['activate_func'] = 'leakly_relu'           # 这个激活函数有问题, 会不收敛
    # R['activate_func'] = 'elu'

    solve_AVE(R)