package com.ai.work;
import com.ai.enmu.ActivationFunction;

import java.util.Arrays;
import java.util.List;

public class IrisMLP {
    public static void main(String[] args) {
        // 定义各层神经元数量，这里假设输入层 4 个（鸢尾花数据 4 个特征），隐藏层 5 个，输出层 3 个（3 个类别）
        int[] layerSizes = {4, 5, 3};
        // 选择激活函数
        ActivationFunction activationFunction = ActivationFunction.SIGMOID;

        // 设置初始学习率
        double initialLearningRate = 0.01;
        // 设置 Adam 优化器的 beta1 参数
        double beta1 = 0.9;
        // 设置 Adam 优化器的 beta2 参数
        double beta2 = 0.999;
        // 设置 Adam 优化器的 epsilon 参数
        double epsilon = 1e-8;
        // 设置 L1 正则化系数
        double l1Lambda = 0.0001;
        // 设置 L2 正则化系数
        double l2Lambda = 0.0001;

        // 创建 MLP 实例
        MLP mlp = new MLP(layerSizes, activationFunction, initialLearningRate, beta1, beta2, epsilon, l1Lambda, l2Lambda);


        List<String[]> rawData = CSVDataReader.readCSV("./iris.csv");
        // 数据预处理
        List<double[]> processedData = DataPreprocessor.preprocessData(rawData);

        // 将处理后的数据转换为二维数组
        double[][] inputs = new double[processedData.size()][4];
        double[][] targets = new double[processedData.size()][3];
        for (int i = 0; i < processedData.size(); i++) {
            double[] data = processedData.get(i);
            System.arraycopy(data, 0, inputs[i], 0, 4);
            int speciesIndex = (int) data[4];
            targets[i][speciesIndex] = 1;
        }

        // 设置训练的轮数
        int epochs = 3000;
        double decayRate = 0.95;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double currentLearningRate = initialLearningRate * Math.pow(decayRate, epoch / 100);
            mlp.setLearningRate(currentLearningRate);
            for (int i = 0; i < inputs.length; i++) {
                mlp.backward(inputs[i], targets[i]);
            }
            if (epoch % 100 == 0) {
                double totalLoss = 0;
                for (int i = 0; i < inputs.length; i++) {
                    totalLoss += mlp.calculateLoss(inputs[i], targets[i]);
                }
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, totalLoss / inputs.length);
            }
        }

        // 计算混淆矩阵
        int[][] confusionMatrix = mlp.calculateConfusionMatrix(inputs, targets);
        System.out.println("Confusion Matrix:");
        for (int[] row : confusionMatrix) {
            System.out.println(Arrays.toString(row));
        }
    }

    // 归一化方法
    public static double[][] normalize(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] normalized = new double[rows][cols];
        double[] min = new double[cols];
        double[] max = new double[cols];

        // 找出每列的最小值和最大值
        for (int j = 0; j < cols; j++) {
            min[j] = Double.MAX_VALUE;
            max[j] = Double.MIN_VALUE;
            for (int i = 0; i < rows; i++) {
                min[j] = Math.min(min[j], data[i][j]);
                max[j] = Math.max(max[j], data[i][j]);
            }
        }

        // 进行归一化处理
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                normalized[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
            }
        }

        return normalized;
    }
}