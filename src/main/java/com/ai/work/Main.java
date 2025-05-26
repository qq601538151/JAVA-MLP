package com.ai.work;
import com.ai.enmu.ActivationFunction;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        try {
            int[] layerSizes = {4, 5, 3};
            ActivationFunction activationFunction = ActivationFunction.SIGMOID;
            double learningRate = 0.01;
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;
            double l1Lambda = 0.01;
            double l2Lambda = 0.01;

            MLP mlp = new MLP(layerSizes, activationFunction, learningRate, beta1, beta2, epsilon, l1Lambda, l2Lambda);

            double[][] inputs = IrisDatasetLoader.loadInputs();
            double[][] targets = IrisDatasetLoader.loadTargets();

            int epochs = 1000;
            for (int epoch = 0; epoch < epochs; epoch++) {
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

            int[][] confusionMatrix = mlp.calculateConfusionMatrix(inputs, targets);
            System.out.println("Confusion Matrix:");
            for (int[] row : confusionMatrix) {
                System.out.println(Arrays.toString(row));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}