package com.ai.work;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class IrisDatasetLoader {
    public static double[][] loadInputs() throws IOException {
        List<double[]> inputsList = new ArrayList<>();
        InputStream inputStream = IrisDatasetLoader.class.getClassLoader().getResourceAsStream("iris.csv");
        if (inputStream == null) {
            throw new IOException("无法找到 iris.csv 文件");
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            if (values.length < 4) {
                // 跳过格式错误的行
                continue;
            }
            double[] input = new double[4];
            for (int i = 0; i < 4; i++) {
                try {
                    input[i] = Double.parseDouble(values[i].trim());
                } catch (NumberFormatException e) {
                    // 处理格式错误的数据
                    System.err.println("无效的数据: " + values[i]);
                    // 可以选择跳过该行或者进行其他处理
                    continue;
                }
            }
            inputsList.add(input);
        }
        reader.close();
        double[][] inputs = new double[inputsList.size()][4];
        for (int i = 0; i < inputsList.size(); i++) {
            inputs[i] = inputsList.get(i);
        }
        return inputs;
    }

    public static double[][] loadTargets() throws IOException {
        List<double[]> targetsList = new ArrayList<>();
        InputStream inputStream = IrisDatasetLoader.class.getClassLoader().getResourceAsStream("iris.csv");
        if (inputStream == null) {
            throw new IOException("无法找到 iris.csv 文件");
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            if (values.length < 5) {
                // 跳过格式错误的行
                continue;
            }
            String species = values[4];
            double[] target = new double[3];
            if (species.equals("Iris-setosa")) {
                target[0] = 1.0;
            } else if (species.equals("Iris-versicolor")) {
                target[1] = 1.0;
            } else if (species.equals("Iris-virginica")) {
                target[2] = 1.0;
            }
            targetsList.add(target);
        }
        reader.close();
        double[][] targets = new double[targetsList.size()][3];
        for (int i = 0; i < targetsList.size(); i++) {
            targets[i] = targetsList.get(i);
        }
        return targets;
    }
}