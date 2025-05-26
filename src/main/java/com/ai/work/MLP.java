package com.ai.work;

import com.ai.enmu.ActivationFunction;

import java.util.Random;

/**
* @Description: MLP 类，实现多层感知机的核心功能
* @Param:
* @return:
* @Author: PK
* @Date: 2025/3/17
*/
class MLP {
    // 存储各层神经元数量的数组
    private int[] layerSizes;
    // 存储各层之间的权重
    // weights[i] 表示第 i 层到第 i+1 层的权重
    private double[][] weights;
    // 存储各层的偏置
    // biases[i] 表示第 i+1 层的偏置
    private double[][] biases;
    // 所使用的激活函数
    private ActivationFunction activationFunction;
    // 学习率，控制每次参数更新的步长
    private double learningRate;
    // Adam 优化器的参数 beta1，用于一阶矩估计的指数衰减率
    private double beta1;
    // Adam 优化器的参数 beta2，用于二阶矩估计的指数衰减率
    private double beta2;
    // Adam 优化器的参数 epsilon，用于防止除零错误
    private double epsilon;
    // L1 正则化系数，用于控制 L1 正则化的强度
    private double l1Lambda;
    // L2 正则化系数，用于控制 L2 正则化的强度
    private double l2Lambda;
    // 权重的一阶矩估计
    private double[][] mWeights;
    // 偏置的一阶矩估计
    private double[][] mBiases;
    // 权重的二阶矩估计
    private double[][] vWeights;
    // 偏置的二阶矩估计
    private double[][] vBiases;
    // 当前迭代次数
    private int t;

    /**
     * MLP 类的构造函数
     * @param layerSizes 各层神经元数量的数组
     * @param activationFunction 激活函数
     * @param learningRate 学习率
     * @param beta1 Adam 优化器的 beta1 参数
     * @param beta2 Adam 优化器的 beta2 参数
     * @param epsilon Adam 优化器的 epsilon 参数
     * @param l1Lambda L1 正则化系数
     * @param l2Lambda L2 正则化系数
     */
    public MLP(int[] layerSizes, ActivationFunction activationFunction, double learningRate, double beta1, double beta2, double epsilon, double l1Lambda, double l2Lambda) {
        // 初始化各成员变量
        this.layerSizes = layerSizes;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.l1Lambda = l1Lambda;
        this.l2Lambda = l2Lambda;
        this.t = 0;

        // 创建一个随机数生成器，用于初始化权重
        Random random = new Random();
        // 初始化权重数组
        weights = new double[layerSizes.length - 1][];
        // 初始化偏置数组
        biases = new double[layerSizes.length - 1][];
        // 初始化权重的一阶矩估计数组
        mWeights = new double[layerSizes.length - 1][];
        // 初始化偏置的一阶矩估计数组
        mBiases = new double[layerSizes.length - 1][];
        // 初始化权重的二阶矩估计数组
        vWeights = new double[layerSizes.length - 1][];
        // 初始化偏置的二阶矩估计数组
        vBiases = new double[layerSizes.length - 1][];

        // 初始化各层的权重和偏置
        for (int i = 0; i < layerSizes.length - 1; i++) {
            // 为当前层到下一层的权重分配空间
            // layerSizes[i] * layerSizes[i + 1] 表示第 i 层到第 i+1 层的权重数量
            weights[i] = new double[layerSizes[i] * layerSizes[i + 1]];
            // 为当前层的偏置分配空间
            // layerSizes[i + 1] 表示第 i+1 层的神经元数量，即偏置数量
            biases[i] = new double[layerSizes[i + 1]];
            // 为权重的一阶矩估计分配空间
            mWeights[i] = new double[layerSizes[i] * layerSizes[i + 1]];
            // 为偏置的一阶矩估计分配空间
            mBiases[i] = new double[layerSizes[i + 1]];
            // 为权重的二阶矩估计分配空间
            vWeights[i] = new double[layerSizes[i] * layerSizes[i + 1]];
            // 为偏置的二阶矩估计分配空间
            vBiases[i] = new double[layerSizes[i + 1]];

            // 使用 Xavier 初始化方法初始化权重
            for (int j = 0; j < weights[i].length; j++) {
                // Xavier 初始化公式，计算权重的初始化范围
                // Math.sqrt(6.0 / (layerSizes[i] + layerSizes[i + 1])) 计算初始化范围
                double limit = Math.sqrt(6.0 / (layerSizes[i] + layerSizes[i + 1]));
                // 在初始化范围内随机生成权重值
                // random.nextDouble() 生成一个 0 到 1 之间的随机数
                // random.nextDouble() * 2 * limit - limit 将随机数映射到 [-limit, limit] 范围内
                weights[i][j] = random.nextDouble() * 2 * limit - limit;
            }
        }
    }
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    /**
     * 前向传播方法，根据输入计算 MLP 的输出
     * @param input 输入数据
     * @return 输出结果
     */
    public double[] forward(double[] input) {
        // 初始化当前层的输出为输入数据
        double[] current = input;
        // 遍历每一层（除输入层）
        for (int i = 0; i < layerSizes.length - 1; i++) {
            // 存储下一层的输出
            double[] next = new double[layerSizes[i + 1]];
            // 计算下一层每个神经元的输出
            for (int j = 0; j < layerSizes[i + 1]; j++) {
                // 初始化神经元的输入总和为偏置值
                double sum = biases[i][j];
                // 遍历当前层的所有神经元，计算加权和
                for (int k = 0; k < layerSizes[i]; k++) {
                    // 计算当前层第 k 个神经元的输出乘以对应的权重，并累加到总和中
                    sum += current[k] * weights[i][j * layerSizes[i] + k];
                }
                // 对总和应用激活函数得到输出
                next[j] = activationFunction.activate(sum);
            }
            // 更新当前层的输出为下一层的输出
            current = next;
        }
        return current;
    }

    /**
     * 反向传播方法，计算梯度并更新权重和偏置
     * @param input 输入数据
     * @param target 目标输出数据
     */
    public void backward(double[] input, double[] target) {
        // 迭代次数加 1
        t++;
        // 获取层数
        int numLayers = layerSizes.length;
        // 存储各层的激活值
        double[][] activations = new double[numLayers][];
        // 存储各层神经元的输入总和
        double[][] zs = new double[numLayers - 1][];

        // 前向传播过程，记录中间值
        activations[0] = input;
        for (int i = 0; i < numLayers - 1; i++) {
            // 为当前层的 z 值分配空间
            zs[i] = new double[layerSizes[i + 1]];
            // 为下一层的激活值分配空间
            activations[i + 1] = new double[layerSizes[i + 1]];
            for (int j = 0; j < layerSizes[i + 1]; j++) {
                // 初始化神经元的输入总和为偏置值
                double sum = biases[i][j];
                for (int k = 0; k < layerSizes[i]; k++) {
                    // 计算当前层第 k 个神经元的输出乘以对应的权重，并累加到总和中
                    sum += activations[i][k] * weights[i][j * layerSizes[i] + k];
                }
                // 记录当前神经元的输入总和
                zs[i][j] = sum;
                // 对总和应用激活函数得到输出
                activations[i + 1][j] = activationFunction.activate(sum);
            }
        }

        // 计算输出层的误差
        double[] delta = new double[layerSizes[numLayers - 1]];
        for (int i = 0; i < layerSizes[numLayers - 1]; i++) {
            // 误差计算公式：(输出 - 目标输出) * 激活函数导数
            delta[i] = (activations[numLayers - 1][i] - target[i]) * activationFunction.derivative(zs[numLayers - 2][i]);
        }

        // 存储各层权重的梯度
        double[][] gradWeights = new double[numLayers - 1][];
        // 存储各层偏置的梯度
        double[][] gradBiases = new double[numLayers - 1][];
        // 反向传播更新梯度
        for (int i = numLayers - 2; i >= 0; i--) {
            // 为当前层的权重梯度分配空间
            gradWeights[i] = new double[layerSizes[i] * layerSizes[i + 1]];
            // 为当前层的偏置梯度分配空间
            gradBiases[i] = new double[layerSizes[i + 1]];
            for (int j = 0; j < layerSizes[i + 1]; j++) {
                // 偏置的梯度等于误差
                gradBiases[i][j] = delta[j];
                for (int k = 0; k < layerSizes[i]; k++) {
                    // 权重的梯度等于误差乘以当前层的激活值
                    gradWeights[i][j * layerSizes[i] + k] = delta[j] * activations[i][k];
                }
            }
            if (i > 0) {
                // 计算上一层的误差
                double[] newDelta = new double[layerSizes[i]];
                for (int j = 0; j < layerSizes[i]; j++) {
                    double sum = 0;
                    for (int k = 0; k < layerSizes[i + 1]; k++) {
                        // 计算上一层误差的加权和
                        sum += delta[k] * weights[i][k * layerSizes[i] + j];
                    }
                    // 乘以激活函数导数得到上一层的误差
                    newDelta[j] = sum * activationFunction.derivative(zs[i - 1][j]);
                }
                // 更新误差
                delta = newDelta;
            }
        }

        // 添加正则化项到权重梯度
        for (int i = 0; i < numLayers - 1; i++) {
            for (int j = 0; j < gradWeights[i].length; j++) {
                // L1 正则化梯度：l1Lambda * sign(权重)
                // L2 正则化梯度：l2Lambda * 权重
                // Math.signum(weights[i][j]) 返回权重的符号
                gradWeights[i][j] += l1Lambda * Math.signum(weights[i][j]) + l2Lambda * weights[i][j];
            }
        }

        // 使用 Adam 优化器更新权重和偏置
        for (int i = 0; i < numLayers - 1; i++) {
            for (int j = 0; j < gradWeights[i].length; j++) {
                // 更新权重的一阶矩估计
                // mWeights[i][j] 乘以 beta1 加上梯度乘以 (1 - beta1)
                mWeights[i][j] = beta1 * mWeights[i][j] + (1 - beta1) * gradWeights[i][j];
                // 更新权重的二阶矩估计
                // vWeights[i][j] 乘以 beta2 加上梯度的平方乘以 (1 - beta2)
                vWeights[i][j] = beta2 * vWeights[i][j] + (1 - beta2) * gradWeights[i][j] * gradWeights[i][j];
                // 计算一阶矩估计的偏差修正值
                // mWeights[i][j] 除以 (1 - beta1^t)
                double mHat = mWeights[i][j] / (1 - Math.pow(beta1, t));
                // 计算二阶矩估计的偏差修正值
                // vWeights[i][j] 除以 (1 - beta2^t)
                double vHat = vWeights[i][j] / (1 - Math.pow(beta2, t));
                // 根据 Adam 优化器公式更新权重
                // 权重减去学习率乘以 mHat 除以 (sqrt(vHat) + epsilon)
                weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
            for (int j = 0; j < gradBiases[i].length; j++) {
                // 更新偏置的一阶矩估计
                mBiases[i][j] = beta1 * mBiases[i][j] + (1 - beta1) * gradBiases[i][j];
                // 更新偏置的二阶矩估计
                vBiases[i][j] = beta2 * vBiases[i][j] + (1 - beta2) * gradBiases[i][j] * gradBiases[i][j];
                // 计算一阶矩估计的偏差修正值
                double mHat = mBiases[i][j] / (1 - Math.pow(beta1, t));
                // 计算二阶矩估计的偏差修正值
                double vHat = vBiases[i][j] / (1 - Math.pow(beta2, t));
                // 根据 Adam 优化器公式更新偏置
                biases[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    /**
     * 计算损失函数（均方误差），并添加正则化项
     * @param input 输入数据
     * @param target 目标输出数据
     * @return 损失值
     */
    public double calculateLoss(double[] input, double[] target) {
        // 进行前向传播得到输出
        double[] output = forward(input);
        // 初始化损失值为 0
        double loss = 0;
        // 计算均方误差
        for (int i = 0; i < output.length; i++) {
            // 计算输出与目标输出差值的平方，并累加到损失值中
            loss += Math.pow(output[i] - target[i], 2);
        }
        // 均方误差除以 2
        loss /= 2;

        // 计算 L1 正则化损失
        double l1Loss = 0;
        // 计算 L2 正则化损失
        double l2Loss = 0;
        for (double[] weight : weights) {
            for (double w : weight) {
                // 计算权重的绝对值之和
                l1Loss += Math.abs(w);
                // 计算权重的平方和
                l2Loss += w * w;
            }
        }
        // 将正则化损失添加到总损失中
        // l1Lambda 乘以 l1Loss 加上 0.5 乘以 l2Lambda 乘以 l2Loss
        loss += l1Lambda * l1Loss + 0.5 * l2Lambda * l2Loss;
        return loss;
    }

    /**
     * 计算混淆矩阵（用于分类任务）
     * @param inputs 输入数据数组
     * @param targets 目标输出数据数组
     * @return 混淆矩阵
     */
    public int[][] calculateConfusionMatrix(double[][] inputs, double[][] targets) {
        // 获取输出层的神经元数量，即类别数量
        int numClasses = layerSizes[layerSizes.length - 1];
        // 初始化混淆矩阵
        int[][] confusionMatrix = new int[numClasses][numClasses];
        for (int i = 0; i < inputs.length; i++) {
            // 进行前向传播得到输出
            double[] output = forward(inputs[i]);
            // 获取预测的类别索引
            int predicted = argmax(output);
            // 获取实际的类别索引
            int actual = argmax(targets[i]);
            // 更新混淆矩阵
            // 实际类别为 actual，预测类别为 predicted 的位置加 1
            confusionMatrix[actual][predicted]++;
        }
        return confusionMatrix;
    }

    /**
     * 辅助方法，用于找到数组中最大值的索引
     * @param array 输入数组
     * @return 最大值的索引
     */
    private int argmax(double[] array) {
        // 初始化最大值的索引为 0
        int maxIndex = 0;
        // 初始化最大值为数组的第一个元素
        double maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                // 如果当前元素大于最大值，更新最大值和最大值的索引
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
