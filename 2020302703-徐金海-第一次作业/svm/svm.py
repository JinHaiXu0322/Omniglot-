# @Author: Jinhai Xu
# @Time: 2023/4/1 2:33
# @IDE: PyCharm
# @Project Name: svm_l -> svm
import cv2                          #导入opencv库
import numpy as np                  #导入numpy库
import matplotlib.pyplot as plt     #导入matplotlib库
from collections import defaultdict #导入defaultdict类，它是字典的子类
import scipy.io as scio             #导入scipy.io库
import keras                        #导入keras库
import pandas as pd

# 分离训练集的图像和标签
def split_data(data):
    X = np.resize(data, [2970, 28, 28])  # 将输入数据data重新变形为大小为[2970, 28, 28]的数组
    label = np.zeros([2970])            # 初始化大小为2970的0数组作为标签
    label_info = np.array([0 for i in range(15)])  # 初始化一个大小为15的数组作为标签信息
    for i in range(198):
        label[i * 15:15 + i * 15] = label_info   # 将每15个图像的标签设为相同的数字
        label_info += 1
    return X, label

# 分离测试集的图像和标签
def split_test(data):
    X = np.resize(data, [990, 28, 28])   # 将输入数据data重新变形为大小为[990, 28, 28]的数组
    label = np.zeros([990])             # 初始化大小为990的0数组作为标签
    label_info = np.array([0 for i in range(5)])  # 初始化一个大小为5的数组作为标签信息
    for i in range(198):
        label[i * 5:5 + i * 5] = label_info     # 将每5个图像的标签设为相同的数字
        label_info += 1
    return X, label

# 导入数据集
dataFile = 'data.mat'
data = scio.loadmat(dataFile)

# train_dataset和train_labels代表训练集的图像与标签, test_dataset与test_labels代表测试集的图像与标签
(train_dataset, train_labels) = split_data(data['train'])   # 将训练集的数据分离为图像和标签
(test_dataset, test_labels) = split_test(data['test'])      # 将测试集的数据分离为图像和标签

SIZE_IMAGE = train_dataset.shape[1]   # 图像的大小为28x28

train_labels = np.array(train_labels, dtype=np.int32)   # 将标签转换为int32类型的numpy数组

# 预处理函数
def deskew(img):
    m = cv2.moments(img)   # 计算输入图像的矩
    if abs(m['mu02']) < 1e-2:  # 判断是否需要进行校正
        return img.copy()
    skew = m['mu11'] / m['mu02']   # 计算图像的倾斜角
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])  # 创建变换
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
#
# # HOG 高级描述符
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
def get_hog():
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    # 初始化HOG描述符
    print("hog descriptor size: {}".format(hog.getDescriptorSize()))
    return hog
'''
# 模型初始化函数 SVM_CHI2, SVM_RBF, SVM_LINEAR, SVM_POLY, SVM_SIGMOID
修改核函数时，需要手动修改model.setKernel(cv2.ml.SVM_CHI2)中的核函数，如上图所示
'''
def svm_init(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()  # 创建SVM模型
    model.setKernel(cv2.ml.SVM_RBF)
    model.setGamma(gamma)  # 设置SVM模型参数
    model.setC(C)
    # model.setCoef0(0.1)
    # 在使用POLY核函数时需要下面的语句设置多项式的次数
    # model.setDegree(3)
    # 在使用NOVA核函数时需要下面的语句设置degree参数
    # model.setDegree(2)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # 设置SVM模型的训练终止条件
    return model

# 模型训练函数，用于训练SVM模型
def svm_train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    # 对SVM模型进行训练
    return model

# 模型预测函数，用于对新样本进行预测
def svm_predict(model, samples):
    return model.predict(samples)[1].ravel()

# 模型评估函数，用于评估SVM模型的分类准确率
def svm_evaluate(model, samples, labels):
    predictions = svm_predict(model, samples)  # 对样本进行预测
    acc = (labels == predictions).mean()  # 计算分类准确率
    return acc * 100

def rotate(image, angle):
    '''
    对图像进行旋转增强
    '''
    rows, cols = image.shape
    # 构造旋转矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # 进行仿射变换
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def shift(image, x, y):
    '''
    对图像进行平移增强
    '''
    rows, cols = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def zoom(image, scale):
    '''
    对图像进行缩放增强
    '''
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def enhance(image):
    '''
    对图像进行一系列增强操作，生成新的增强图像
    '''
    images = []
    angles = [-10, -5, 5, 10]  # 旋转角度列表
    shifts = [(5, 5), (-5, -5), (5, -5), (-5, 5)]  # 平移距离列表
    scales = [0.9, 1.1]  # 缩放比例列表

    # 旋转增强
    for angle in angles:
        img = rotate(image, angle)
        images.append(img)
    # 平移增强
    for shift_x, shift_y in shifts:
        img = shift(image, shift_x, shift_y)
        images.append(img)
    # 缩放增强
    for scale in scales:
        img = zoom(image, scale)
        images.append(img)
    return images
# 对训练集图像进行增强
augmented_images = []
augmented_labels = []
for i in range(len(train_dataset)):
    image = train_dataset[i]
    label = train_labels[i]
    augmented_images.append(image)
    augmented_labels.append(label)
    enhanced_images = enhance(image)
    for enhanced_image in enhanced_images:
        augmented_images.append(enhanced_image)
        augmented_labels.append(label)

# 转化为numpy数组
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# 数据打散，将原来顺序排列的图像打乱，增加随机性
shuffle = np.random.permutation(len(augmented_images))
augmented_images, augmented_labels = augmented_images[shuffle], augmented_labels[shuffle]
# # 数据打散，将原来顺序排列的图像打乱，增加随机性
# shuffle = np.random.permutation(len(train_dataset))
# train_dataset, train_labels = train_dataset[shuffle], train_labels[shuffle]

# 使用 HOG 描述符
hog = get_hog()
hog_descriptors = []
for img in augmented_images:
    # hog_descriptors.append(hog.compute(deskew(img)))
    img = cv2.convertScaleAbs(img)
    hog_descriptors.append(hog.compute(img))
hog_descriptors = np.squeeze(hog_descriptors)

# 训练数据与测试数据划分
partition = int(0.9 * len(hog_descriptors))
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
labels_train, labels_test = np.split(augmented_labels, [partition])

print('Training SVM model ...')
results = defaultdict(list)

result_df = pd.DataFrame(columns=["C", "gamma", "accuracy"])

result = []

for C in [0.1, 1, 3, 5, 8, 12, 20, 40, 50, 100]:
    results[C] = []
    for gamma in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1]:
        model = svm_init(C, gamma)
        svm_train(model, hog_descriptors_train, labels_train)
        testacc = svm_evaluate(model, hog_descriptors_test, labels_test)
        print("C = %.1f , gamma = %.4f" % (C, gamma))
        print(" {}".format("accuracy = %.2f" % testacc))
        result_df = pd.concat([result_df, pd.DataFrame({"C": [C], "gamma": [gamma], "accuracy": [testacc]})], ignore_index=True)
        results[C].append(testacc)
        result.append(testacc)
result_df.to_excel("RBF_result.xlsx", index=False)
result.sort(reverse=True)
print(result)


# 可视化结果
fig = plt.figure(figsize=(10, 6))
plt.suptitle("SVM WITH RBF KERNEL", fontsize=16, fontweight='bold')
ax = plt.subplot(1,1,1)
ax.set_xlim(0, 0.5)
dim = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1]

for key in results:
    ax.plot(dim, results[key], linestyle='--', marker='o', label=str(key))
plt.legend(loc='upper right', title="C")
plt.title('Accuracy of the SVM model varying both C and gamma')
plt.xlabel("gamma")
plt.ylabel("accuracy")
plt.show()