import random
import keras.backend as backend
import skimage.io as io
import os
import math
import keras
import numpy as np
import foolbox as fb
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
from scipy import stats

tf.compat.v1.disable_eager_execution()

def load_pic(load_path, dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        img_arr = []
        name_arr = []
        label_arr = []
        class_path = os.path.join(load_path, str(i))
        file_list = os.listdir(class_path)
        for file in file_list:
            file_path = os.path.join(class_path, file)
            img = io.imread(file_path)
            # temp_label = [0,0,0,0,0,0,0,0,0,0]
            # temp_label[i] = 1
            temp_label = np.array([i])
            if dataset == 'mnist':
                img = img[:, :, 0]
                img = np.expand_dims(img, axis=0)
                img = img.transpose((1, 2, 0))
            else:
                img = img
                # print(img.shape)
            img_arr.append(img)
            name_arr.append(file)
            label_arr.append(temp_label)
        label_arr = np.array(label_arr)
        img_arr = np.array(img_arr)
        name_arr = np.array(name_arr)
        print(label_arr.shape)
        print(img_arr.shape)
        print(name_arr.shape)
        np.save(os.path.join(save_path, str(i) + '_imgs.npy'), img_arr)
        np.save(os.path.join(save_path, str(i) + '_names.npy'), name_arr)
        np.save(os.path.join(save_path, str(i) + '_labels.npy'), label_arr)
        print(str(i), ":finish!")

def get_adv_pic(model_name, temp_arr, label, num):
    with tf.Session() as sess:
        model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model_name, model_name + '.h5')
        model = load_model(model_path)
        bounds = (0, 1)
        fool_model = fb.models.TensorFlowModel.from_keras(model, bounds=bounds, preprocessing=(0, 1))
        attack = fb.attacks.SaliencyMapAttack(fool_model)
        adv = attack(inputs=temp_arr, labels=label, num_random_targets=num)
        # print(model.predict(adv))
        index = np.argmax(model.predict(adv))
    backend.clear_session()
    return adv, index

def gen_adv_pic(load_path, save_path, dataset, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        if i != 0:
            continue
        class_path = os.path.join(save_path, str(i))
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        label = np.array([i])
        ori_img = np.load(os.path.join(load_path, str(i) + '_imgs.npy'))
        ori_img = ori_img.astype('float32') / 255.0
        name_arr = np.load(os.path.join(load_path, str(i) + '_names.npy'))
        for j in range(len(ori_img)):
            if j < 492:
                continue
            print('class:', str(i), ', pic:', name_arr[j], ',num:', str(j), ', start!')
            flag = np.zeros(10, dtype='uint8')
            flag[i] = 1
            if dataset == 'mnist':
                temp_arr = np.empty((1, 28, 28, 1), dtype='float32')
                temp_arr[0] = ori_img[j]
                save_arr = np.empty((10, 28, 28, 1), dtype='float32')
            temp_num = 0
            while(not flag.all()):
                # print(flag)
                # print(temp_num)
                # temp_num += 1
                num = random.randint(0, 9) % 10
                # print(num)
                adv, index = get_adv_pic(model_name, temp_arr, label, num)
                # print(index)
                if flag[index] == 0:
                    flag[index] = 1
                    save_arr[index] = adv[0]
                    temp_num = 0
                else:
                    temp_num += 1
                if temp_num == 10:
                    break
            for k in range(len(save_arr)):
                if k == i or flag[k] == 0:
                    continue
                dir_path = os.path.join(class_path, str(k))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                pic_path = os.path.join(dir_path, name_arr[j])
                img = (save_arr[k] * 255).astype('uint8')
                io.imsave(pic_path, img)

def search_pic(a_img, b_img, model, i, j):
    t = 0
    beta = 0.0001
    a = a_img.astype('float32') / 255.0
    a = np.expand_dims(a, axis=0)
    a = a.transpose((1, 2, 0))
    a = np.expand_dims(a, axis=0)
    b = b_img.astype('float32') / 255.0
    b = b[:, :, 0]
    b = np.expand_dims(b, axis=0)
    b = b.transpose((1, 2, 0))
    b = np.expand_dims(b, axis=0)
    c = (a + b) / 2.0
    c_output = model.predict(c)[0]
    c_i_output = c_output[i]
    c_j_output = c_output[j]
    while math.fabs(c_i_output - c_j_output) > beta:
        t += 1
        if c_i_output > c_j_output:
            b = c
        else:
            a = c
        c = (a + b) / 2.0
        c_output = model.predict(c)[0]
        c_i_output = c_output[i]
        c_j_output = c_output[j]
        if t == 10000:
            c = np.zeros((1, 28, 28, 1), dtype='float')
            break
    return c

def find_boundary_pic(a_load_path, b_load_path, dataset, save_path, model):
    model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model, model + '.h5')
    model = load_model(model_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        a_class_path = os.path.join(a_load_path, str(i))
        c_class_path = os.path.join(save_path, str(i))
        for j in range(10):
            if j == i:
                continue
            a_boundary_class = os.path.join(a_class_path, str(j))
            a_file_list = os.listdir(a_boundary_class)
            c_boundary_class = c_class_path + '_' + str(j) + '.npy'
            temp_num = 0
            print('class:', str(i), ', boundary:', str(j))
            for file in tqdm(a_file_list):
                temp_arr = np.empty((1, 28, 28, 1), dtype='float32')
                a_file_path = os.path.join(a_boundary_class, file)
                b_file_path = os.path.join(b_load_path, str(i), file)
                a_img = io.imread(a_file_path)
                b_img = io.imread(b_file_path)
                c_img = search_pic(a_img, b_img, model, i, j)
                if not c_img.any():
                    print(file, 'error!')
                    continue
                c_img = c_img[0]
                temp_arr[0] = c_img
                if temp_num == 0:
                    file_arr = temp_arr
                else:
                    file_arr = np.concatenate((file_arr, temp_arr), axis=0)
                temp_num += 1
            print(file_arr.shape)
            np.save(c_boundary_class, file_arr)

def save_boundary_pic_npy(load_path, save_path, model):
    model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model, model + '.h5')
    model = load_model(model_path)
    for i in range(10):
        for j in range(10):
            coordinate = []
            if j == i:
                continue
            temp_arr_1 = np.load(os.path.join(load_path, str(i) + '_' + str(j) + '.npy'))
            temp_arr_2 = np.load(os.path.join(load_path, str(j) + '_' + str(i) + '.npy'))
            temp_arr = np.concatenate((temp_arr_1, temp_arr_2), axis=0)
            outputs = model.predict(temp_arr)
            x = outputs[:, i].reshape(-1)
            y = outputs[:, j].reshape(-1)
            coordinate.append(x)
            coordinate.append(y)
            coordinate = np.array(coordinate)
            print(coordinate.shape)
            np.save(os.path.join(save_path, str(i) + '_' + str(j) + '_boundary_coordinate.npy'), coordinate)

def copy_pic(pic_path, save_path):
    img = io.imread(pic_path)
    io.imsave(save_path, img)

def load_pic1(dataset, pic_path):
    if dataset == 'mnist':
        img = io.imread(pic_path)
        img = img[:, :, 0]
        img = np.expand_dims(img, axis=0)
        img = img.transpose(1, 2, 0)
    else:
        img = io.imread(pic_path)
    return img

def gen_adv_pic1(load_path, class_name, save_path, dataset, model_name, attack_name):
    with tf.Session() as sess:
        model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model_name, model_name + '.h5')
        model = load_model(model_path)
        bounds = (0, 1)
        fool_model = fb.models.TensorFlowModel.from_keras(model, bounds=bounds, preprocessing=(0, 1))
        if attack_name == 'cw':
            attack = fb.attacks.L2BasicIterativeAttack(fool_model)
        elif attack_name == 'fgsm':
            attack = fb.attacks.GradientSignAttack(fool_model)
        elif attack_name == 'bim':
            attack = fb.attacks.L1BasicIterativeAttack(fool_model)
        elif attack_name == 'jsma':
            attack = fb.attacks.SaliencyMapAttack(fool_model)
        else:
            print('对抗生成名字错误，不生成图片！！！')
        if dataset == 'mnist':
            img = load_pic1(dataset, load_path)
            img_arr = np.empty((1, 28, 28, 1), dtype='uint8')
            img_arr[0] = img
            img_arr = img_arr.astype('float32') / 255
            label = np.array([class_name])
        else:
            img = load_pic1(dataset, load_path)
            img_arr = np.empty((3, 32, 32, 3), dtype='uint8')
            img_arr[0] = img
            img_arr = img_arr.astype('float32') / 255
            label = np.array([class_name])
        if attack_name == 'fgsm':
            adv = attack(img_arr, label, [0.01, 0.1])
        else:
            adv = attack(img_arr, label)
        if dataset == 'mnist':
            new_img = np.zeros((28, 28, 3), dtype='uint8')
            for i in range(3):
                new_img[:, :, i] = (adv[0][:, :, 0] * 255.0).astype('uint8')
        else:
            new_img = (adv[0] * 255.0).astype('uint8')
        io.imsave(save_path, new_img)
    backend.clear_session()
    return adv[0]

def select_ori_data(load_path, data_path, num, dataset):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    save_pic_path = data_path + os.sep + 'pic'
    save_npy_path = data_path + os.sep + 'npy'
    if not os.path.exists(save_pic_path):
        os.mkdir(save_pic_path)
    if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)
    for i in range(10):
        class_path = load_path + os.sep + str(i)
        save_class_path = save_pic_path + os.sep + str(i)
        if not os.path.exists(save_class_path):
            os.mkdir(save_class_path)
        pic_list = os.listdir(class_path)
        if dataset == 'mnist':
            pic_arr = np.empty((num, 28, 28, 1))
            name_arr = np.empty(num, dtype='U20')
        else:
            pic_arr = np.empty((num, 32, 32, 3))
            name_arr = np.empty(num, dtype='U20')
        for j in range(num):
            temp_num = random.randint(0, len(pic_list)-1)
            while pic_list[temp_num] in name_arr:
                temp_num = random.randint(0, len(pic_list)-1)
            pic_path = class_path + os.sep + pic_list[temp_num]
            pic_arr[j] = load_pic1(dataset, pic_path)
            name_arr[j] = pic_list[temp_num]
            save_path = save_class_path + os.sep + pic_list[temp_num]
            copy_pic(pic_path, save_path)
        pic_arr = pic_arr.astype('float32') / 255.0
        npy_path = save_npy_path + os.sep + str(i) + '_pictures.npy'
        np.save(npy_path, pic_arr)
        npy_path = save_npy_path + os.sep + str(i) + '_names.npy'
        np.save(npy_path, name_arr)
        print(pic_arr.shape)

def get_adv_data(ori_path, data_path, adv_name, rate, num, dataset, model_name):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    for r in range(len(rate)):
        if r == 0:
            continue
        print(str(rate[r]), ':')
        path = os.path.join(data_path, str(rate[r]))
        if not os.path.exists(path):
            os.mkdir(path)
        save_pic_path = path + os.sep + 'pic'
        save_npy_path = path + os.sep + 'npy'
        if not os.path.exists(save_pic_path):
            os.mkdir(save_pic_path)
        if not os.path.exists(save_npy_path):
            os.mkdir(save_npy_path)
        for i in range(10):
            if r == 1 and i < 3:
                continue
            path2 = os.path.join(save_pic_path, str(i))
            if not os.path.exists(path2):
                os.mkdir(path2)
            if r == 0:
                adv_num = int(num * (rate[r]))
                adv_names = np.empty(adv_num, dtype='U20')
                path1 = os.path.join(ori_path, 'pic', str(i))
                name_arr = np.load(os.path.join(ori_path, 'npy', str(i) + '_names.npy'))
            else:
                adv_num = int(num * (rate[r] - rate[r - 1]))
                temp_names = np.empty(adv_num)
                adv_names = np.load(os.path.join(data_path, str(rate[r-1]), 'npy', str(i) + '_adv_names.npy'))
                adv_names = np.concatenate((temp_names, adv_names), axis=0)
                path1 = os.path.join(data_path, str(rate[r-1]), 'pic', str(i))
                name_arr = np.load(os.path.join(data_path, str(rate[r-1]), 'npy', str(i) + '_names.npy'))
            pic_list = os.listdir(path1)
            for j in range(adv_num):
                temp_num = random.randint(0, len(pic_list) - 1)
                while pic_list[temp_num] in adv_names:
                    temp_num = random.randint(0, len(pic_list) - 1)
                adv_names[j] = pic_list[temp_num]
            if dataset == 'mnist':
                img_arr = np.empty((num, 28, 28, 1), dtype='float32')
            else:
                img_arr = np.empty((num, 32, 32, 3), dtype='float32')
            img_num = 0
            for pic in tqdm(pic_list):
                load_path = os.path.join(path1, pic)
                if pic in adv_names:
                    img = gen_adv_pic1(load_path, i, os.path.join(path2, pic), dataset, model_name, adv_name)
                else:
                    copy_pic(load_path, os.path.join(path2, pic))
                    img = load_pic1(dataset, load_path)
                    img = img.astype('float32') / 255.0
                img_arr[img_num] = img
                img_num += 1
                # print(img_arr[0])
            npy_path = save_npy_path + os.sep + str(i) + '_pictures.npy'
            np.save(npy_path, img_arr)
            npy_path = save_npy_path + os.sep + str(i) + '_names.npy'
            np.save(npy_path, name_arr)
            npy_path = save_npy_path + os.sep + str(i) + '_adv_names.npy'
            np.save(npy_path, adv_names)
            # print(str(i), adv_names)

def save_npy(path, name_path, dataset):
    for i in range(10):
        class_path = os.path.join(path, str(i))
        pic_names = np.load(os.path.join(name_path, str(i) + '_names.npy'))
        if dataset == 'mnist':
            img_arr = np.empty((len(pic_names), 28, 28, 1))
            for j in range(len(pic_names)):
                pic_path = os.path.join(class_path, pic_names[j])
                img = io.imread(pic_path)
                img = img[:, :, 0]
                img = np.expand_dims(img, axis=0)
                img = img.transpose(1, 2, 0)
                img_arr[j] = img
        img_arr = img_arr.astype('float32') / 255
        pic_save_path = os.path.join(name_path, str(i) + '_pictures.npy')
        np.save(pic_save_path, img_arr)

def cal_score(output, i, k, coordinate_path):
    train_coordinates = np.load(os.path.join(coordinate_path, str(i) + '_' + str(k) + '_boundary_coordinate.npy'))
    test_coordinate = np.empty((2, 1), dtype='float32')
    test_coordinate[0][0] = output[i]
    test_coordinate[1][0] = output[k]
    kde = stats.gaussian_kde(train_coordinates, bw_method='scott')
    score = -kde.logpdf(test_coordinate)
    # print(score[0] / 10000000000.0)
    return score[0] / 10000000000.0

def cal_ori_test_data_ked(dataset, model, load_path, save_path, path):
    model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model, model + '.h5')
    model = load_model(model_path)
    for i in range(10):
        print(str(i), ":")
        npy_path = os.path.join(load_path, str(i) + '_pictures.npy')
        pic = np.load(npy_path)
        test_outputs = model.predict(pic)
        scores = []
        for j in range(len(test_outputs)):
            score = 0
            for k in range(10):
                if k == i:
                    continue
                temp_score = cal_score(test_outputs[j], i, k, path)
                score += temp_score
            score = score / 9.0
            scores.append(score)
        scores = np.array(scores)
        print(scores.shape)
        score_path = os.path.join(save_path, str(i) + '_KED_scores.npy')
        np.save(score_path, scores)

def cal_adv_test_data_ked(dataset, model, load_path, save_path, path, rate):
    model_path = os.path.join('D:\PycharmProject\DeepBoundary\model', model, model + '.h5')
    model = load_model(model_path)
    for r in range(len(rate)):
        state_path = os.path.join(save_path, str(rate[r]))
        if not os.path.exists(state_path):
            os.mkdir(state_path)
        for i in range(10):
            print(str(i), ':')
            npy_path = os.path.join(load_path, str(rate[r]), 'npy', str(i) + '_pictures.npy')
            pic = np.load(npy_path)
            test_outputs = model.predict(pic)
            scores = []
            for j in range(len(test_outputs)):
                score = 0
                for k in range(10):
                    if k == i:
                        continue
                    temp_score = cal_score(test_outputs[j], i, k, path)
                    score += temp_score
                score = score / 9.0
                scores.append(score)
            scores = np.array(scores)
            print(scores.shape)
            score_path = os.path.join(state_path, str(i) + '_KED_scores.npy')
            np.save(score_path, scores)

def com_coverage(dataset, load_path, save_path, state, rate):
    for s in range(len(state)):
        print(state[s])
        if s == 0:
            path1 = os.path.join(load_path, state[s])
            path2 = os.path.join(save_path, state[s] + '_coverage.npy')
            for i in range(10):
                temp_score = np.load(os.path.join(path1, str(i) + '_KED_scores.npy'))
                if i == 0:
                    scores = temp_score
                else:
                    scores = np.concatenate((scores, temp_score), axis=0)
            scores = scores * 10000
            temp = np.zeros(2000)
            for i in range(len(scores)):
                t = math.ceil(scores[i])
                if t < 0:
                    t = -t
                if t > 2000:
                    print(scores[i], "惊喜度爆炸")
                    continue
                else:
                    temp[t] = 1
            num1 = sum(temp == 1)
            coverage = num1 / 2500.0
            cover_arr = []
            cover_arr.append(coverage)
            cover_arr = np.array(cover_arr)
            np.save(path2, cover_arr)
            print(cover_arr)
        else:
            for r in range(len(rate)):
                print(rate[r])
                path1 = os.path.join(load_path, state[s], str(rate[r]))
                path2 = os.path.join(save_path, state[s] + '_' + str(rate[r]) + '_coverage.npy')
                for i in range(10):
                    temp_score = np.load(os.path.join(path1, str(i) + '_KED_scores.npy'))
                    if i == 0:
                        scores = temp_score
                    else:
                        scores = np.concatenate((scores, temp_score), axis=0)
                for i in range(10):
                    temp_score = np.load(os.path.join(load_path, state[0], str(i) + '_KED_scores.npy'))
                    scores = np.concatenate((scores, temp_score), axis=0)
                print(scores)
                scores = scores * 10000
                temp = np.zeros(2500)
                for i in range(len(scores)):
                    t = math.ceil(scores[i])
                    if t < 0:
                        t = -t
                    if t > 2000:
                        print(scores[i], "惊喜度爆炸")
                        continue
                    else:
                        temp[t] = 1
                print(temp)
                num1 = sum(temp == 1)
                coverage = num1 / 2000.0
                cover_arr = []
                cover_arr.append(coverage)
                cover_arr = np.array(cover_arr)
                np.save(path2, cover_arr)
                print(cover_arr)


if __name__ == '__main__':
    dataset = 'mnist'
    model = 'lenet1'
    # mddel = 'lenet5'
    load_path = os.path.join('D:\PycharmProject\DeepBoundary\dataset', dataset, 'picture', 'train')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_1_save_pic_npy')
    load_pic(load_path, dataset, save_path)
    #
    load_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_1_save_pic_npy')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_2_get_adv_npy')
    gen_adv_pic(load_path, save_path, dataset, model)
    #
    a_load_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_2_get_adv_npy')
    b_load_path = os.path.join('D:\PycharmProject\DeepBoundary', 'dataset', dataset, 'picture', 'train')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_3_save_boundary_pic')
    find_boundary_pic(a_load_path, b_load_path, dataset, save_path, model)
    #
    load_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_3_save_boundary_pic')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_3_save_boundary_pic')
    save_boundary_pic_npy(load_path, save_path, model)
    #
    path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_4_save_pic_npy')
    load_path = os.path.join('D:\PycharmProject\DeepBoundary\dataset', dataset, 'picture', 'test')
    if not os.path.exists(path):
        os.mkdir(path)
    num = 800
    state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    rate = [0.3, 0.6, 0.9]
    for i in range(len(state)):
        print(state[i], ':')
        data_path = path + os.sep + state[i]
        if i == 0:
            continue
            select_ori_data(load_path, data_path, num, dataset)
        else:
            ori_path = path + os.sep + state[0]
            get_adv_data(ori_path, data_path, state[i], rate, num, dataset, model)
    for i in range(len(state)):
        print(state[i], ':')
        data_path = path + os.sep + state[i]
        if i == 0:
            pic_path = os.path.join(data_path, 'pic')
            name_path = os.path.join(data_path, 'npy')
            save_npy(pic_path, name_path, dataset)
        else:
            for j in range(len(rate)):
                print(rate[j])
                pic_path = os.path.join(data_path, str(rate[j]), 'pic')
                name_path = os.path.join(data_path, str(rate[j]), 'npy')
                save_npy(pic_path, name_path, dataset)
    load_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_4_save_pic_npy')
    path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_3_save_boundary_pic')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_5_cal_test_data_KED')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    rate = [0.3, 0.6, 0.9]
    for i in range(len(state)):
        print(state[i], ':')
        data_path = load_path + os.sep + state[i]
        save_state_path = os.path.join(save_path, state[i])
        if not os.path.exists(save_state_path):
            os.mkdir(save_state_path)
        if i == 0:
            name_path = os.path.join(data_path, 'npy')
            cal_ori_test_data_ked(dataset, model, name_path, save_state_path, path)
        else:
            continue
            cal_adv_test_data_ked(dataset, model, data_path, save_state_path, path, rate)

    load_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_5_cal_test_data_KED')
    save_path = os.path.join('D:\PycharmProject\DeepBoundary\data', model, 'step_6_cal_coverage')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    state = ['ori', 'fgsm', 'bim', 'jsma', 'cw']
    rate = [0.3, 0.6, 0.9]
    com_coverage(dataset, load_path, save_path, state, rate)
