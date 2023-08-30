import os
import shutil
import io
import signal
from contextlib import redirect_stdout

import argparse


##################### Utils #########################


def write_output(output, fpath):
    f = open(fpath, "w")
    f.write(str(output))
    f.close()


def mkdir(name, rm=True):
    if not os.path.exists(name):
        os.makedirs(name)
    elif rm:
        shutil.rmtree(name)
        os.makedirs(name)


class TimeOutException(Exception):
    pass


def handler(signum, frame):
    raise TimeOutException()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw7')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ###################################
    #             Grade               #
    ###################################

    report_path = 'scores.txt'
    output_path = 'output'
    mkdir(output_path)
    freport = open(report_path, 'w')
    signal.signal(signal.SIGALRM, handler)

    fnames = ['part1', 'part2', 'part3']
    fnames_description = ['Part 1: Creating LeNet',
                          'Part 2: Calculating the model parameters',
                          'Part 3: Training under different configurations']

    ntests = 3
    model_path = './model_best.pth.tar'

    total_score = 0
    for i in range(1, ntests + 1):
        if args.single and i != int(args.test):
            continue
        out = ''
        score = 0
        message = ''

        if True:
            try:
                if i == 1:
                    temp1 = 0
                    from student_code import LeNet
                    import torch

                    model = LeNet()
                    try:
                        # breakpoint()
                        gt = {1: [],
                              2: [2, 16, 5, 5],
                              3: [],
                              4: [],
                              5: [],
                              6: []}
                        _, output = model(torch.rand(2, 3, 32, 32))
                        print(output)
                    except:
                        score = 0
                        temp1 = 1
                        message += '\nYour model cannot accept the input of the required dataset.\n'
                        message += '\nScore for Part 1 is 0.\n'
                    if temp1 != 1:
                        try:
                            # if output[1][0] == gt[1][0] and output[1][1] == gt[1][1] and \
                            #         output[1][2] == gt[1][2] and output[1][3] == gt[1][3]:
                            #     score += 5
                            if output[2][0] == gt[2][0] and output[2][1] == gt[2][1] and \
                                    output[2][2] == gt[2][2] and output[2][3] == gt[2][3]:
                                score += 30
                            # if output[3][0] == gt[3][0] and output[3][1] == gt[3][1]:
                            #     score += 5
                            # if output[4][0] == gt[4][0] and output[4][1] == gt[4][1]:
                            #     score += 5
                            # if output[5][0] == gt[5][0] and output[5][1] == gt[5][1]:
                            #     score += 5
                            # if output[6][0] == gt[6][0] and output[6][1] == gt[6][1]:
                            #     score += 5
                            message += '\nScore less than 30 means missing keys or wrong shape existed.\n'
                        except:
                            message += '\nUnexpected keys or no keys in the returned dict.\n'
                        message += '\nScore for Part 1 is ' + str(score) + '\n'
                        message += '\nYour output is:\n'
                        message += str(output)
                        message += '\n'
                    out = output


                elif i == 2:
                    from student_code import count_model_params
                    import torch

                    params = count_model_params()
                    print(params)
                    if params < 0.15 or params > 0.16:
                        score = 0
                        message += '\nParamters are not calculated in a right range!\n'
                    else:
                        score += 20
                    message += '\nScore for Part 2 is ' + str(score) + '\n'
                    out = params
                    message += '\nYour output is:\n'
                    message += str(out)
                    message += '\nThe desired output is in 0.15 to 0.16\n'

                elif i == 3:
                    f = open("results.txt", "r")
                    files = f.readlines()
                    try:
                        if float(files[6].strip()) > 12.85 and float(files[6].strip()) < 14.85:
                            score += 50
                    except:
                        message += '\nThe results in the .txt file are not complete.\n'
                    message += '\nScore for Part 3 is ' + str(score) + '\n'
                    message += '\nYour output 6 is:\n'
                    files_new = float(files[6].strip())
                    message += str(files_new)
                    message += '\nThe desired output 6 (bigger or less than this one within 1% range is acceptable) is:\n'
                    message += str([13.85])
                    message += '\n'
                total_score += score
            except TimeOutException as exc:
                message = "Time Out"
            except ImportError:
                message = "Function is NOT found"
            except Exception as e:
                message = "Exception: " + str(e)

            mess = "{}. {} {}\n".format(i, fnames_description[i - 1], message)
            if args.single and int(args.test) == i:
                print(mess)
                print('Output: ', out)
                print('Score: ', score)
            else:
                if not args.single:
                    print(mess)
                freport.write(mess)
                write_output(out, os.path.join(
                    output_path, '{}.txt'.format(fnames[i - 1])))

    if not (args.single or args.debug):
        print('===> score: {}'.format(total_score))
        freport.write('Total: {}/100'.format(total_score))
        freport.close()