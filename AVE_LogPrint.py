import AVE_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    AVE_tools.log_string('The type of problem: %s\n' % str(R_dic['problem_type']), log_fileout)
    AVE_tools.log_string('The name of equation: %s\n' % str(R_dic['eqs_name']), log_fileout)
    AVE_tools.log_string('The dim-row and dim-column of equation: %s and %s\n' % (str(R_dic['Row']), str(R_dic['Column'])), log_fileout)
    AVE_tools.log_string('Network model for solving problem: %s\n' % str(R_dic['model']), log_fileout)
    AVE_tools.log_string('The hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if str(R_dic['model']) == "AVE_DNN_Fourier":
        AVE_tools.log_string('Activate function of input for solving problem: %s\n' % '[sin;cos]', log_fileout)
    else:
        AVE_tools.log_string('Activate function of input for solving problem: %s\n' % str(R_dic['activate_func2in']), log_fileout)

    AVE_tools.log_string('Activate function for solving problem: %s\n' % str(R_dic['activate_func']), log_fileout)

    AVE_tools.log_string('Activate function of output for solving problem: %s\n' % str(R_dic['activate_func2out']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        AVE_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        AVE_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    AVE_tools.log_string('The initial learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    AVE_tools.log_string('The decay for learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)

    AVE_tools.log_string('The model of regularizing weights: %s\n' % str(R_dic['regular_weight_model']), log_fileout)
    if R_dic['sqrt_error'] == 1:
        AVE_tools.log_string('The model of solution error and function error: %s\n' % str('sqrt_error'), log_fileout)
    else:
        AVE_tools.log_string('The model of solution error and function error: %s\n' % str('square_error'), log_fileout)

    if R_dic['activate_stop'] != 0:
        AVE_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        AVE_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                             log_fileout)


def log_Print2one_epoch(i_epoch, run_time, tmp_lr, penalty2func, pwb, sol_error, sol_rel, loss, fun_error, func_res,
                        log_out=None):
    # 将运行结果打印出来
    print('train epoch: %d, time: %.5f' % (i_epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty of function error: %.10f' % penalty2func)
    print('weights and biases with penalty: %.10f' % pwb)
    print('total loss for training: %.12f' % loss)
    print('solution error for training: %.15f' % sol_error)
    print('solution residual for training: %.15f' % sol_rel)
    print('function error for training: %.15f' % fun_error)
    print('function residual error for training: %.15f\n' % func_res)

    AVE_tools.log_string('train epoch: %d,time: %.5f' % (i_epoch, run_time), log_out)
    AVE_tools.log_string('learning rate: %.10f' % tmp_lr, log_out)
    AVE_tools.log_string('penalty of function error: %.10f' % penalty2func, log_out)
    AVE_tools.log_string('weights and biases with penalty: %.10f' % pwb, log_out)
    AVE_tools.log_string('total loss for training: %.12f' % loss, log_out)
    AVE_tools.log_string('solution error for training: %.15f' % sol_error, log_out)
    AVE_tools.log_string('solution residual for training: %.15f' % sol_rel, log_out)
    AVE_tools.log_string('function error for training: %.15f' % fun_error, log_out)
    AVE_tools.log_string('function residual error for training: %.15f\n' % func_res, log_out)
