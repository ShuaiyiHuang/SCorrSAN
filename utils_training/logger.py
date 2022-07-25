import os
import logging
import time
import datetime

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.name_exp

        #cls.logpath = os.path.join(args.logroot, logpath + logtime + '.log')
        cls.logpath = os.path.join(args.snapshots, logpath)
        cls.benchmark = args.benchmark

        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        def setup_logger(logger_name,log_file, mode='a', level=logging.INFO):
            # mode 'a' or 'w'
            l = logging.getLogger(logger_name)
            formatter = logging.Formatter('%(message)s')
            #fileHandler = logging.FileHandler(log_file,mode='w')
            fileHandler = logging.FileHandler(log_file,mode=mode) # append mode
            fileHandler.setFormatter(formatter)
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)

            l.setLevel(level)
            l.addHandler(fileHandler)
            l.addHandler(streamHandler)

        exp_prefix = '_'.join(args.name_exp.split('_')[:2])
    
        setup_logger(logger_name='log_args',log_file=os.path.join(cls.logpath, 'log_args_{}.txt'.format(exp_prefix)),mode='w')
        setup_logger(logger_name='log_all',log_file=os.path.join(cls.logpath, 'log_all_{}.txt'.format(exp_prefix)), mode='a')
        setup_logger(logger_name='log_main', log_file=os.path.join(cls.logpath, 'log_main_{}.txt'.format(exp_prefix)), mode='a')
       
        logger_args = logging.getLogger('log_args')
        logger_all = logging.getLogger('log_all')
        logger_main = logging.getLogger('log_main')

        cls.file_log_all = os.path.join(cls.logpath, 'log_all_{}.txt'.format(exp_prefix))

        # Log arguments
        logger_args.info('\n+=========== {} ============+'.format('SCorrSAN'))
        for arg_key in args.__dict__:
            logger_args.info('%s:%s' % (arg_key, str(args.__dict__[arg_key])))
        logger_args.info('+================================================+\n')

        cls.logger_all = logger_all
        cls.logger_main = logger_main

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt and print"""
        cls.logger_all.info(msg)

    @classmethod
    def info_main(cls, msg):
        r"""Writes message to .txt for main results and print"""
        cls.logger_main.info(msg)

    @classmethod
    def info_np(cls, msg):
        # r"""Writes message to .txt w/o print"""
        outF = open(cls.file_log_all, "a")

        outF.write(msg)
        outF.write("\n")
        outF.close()
    
