package com.ai.work;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DataPreprocessor {
    public static List<double[]> preprocessData(List<String[]> rawData) {
        List<double[]> processedData = new ArrayList<>();
        Map<String, Integer> speciesMap = new HashMap<>();
        int speciesIndex = 0;

        // 跳过表头
        for (int i = 1; i < rawData.size(); i++) {
            String[] row = rawData.get(i);
            double[] data = new double[5];
            for (int j = 1; j < row.length - 1; j++) {
                data[j - 1] = Double.parseDouble(row[j]);
            }
            String species = row[row.length - 1];
            if (!speciesMap.containsKey(species)) {
                speciesMap.put(species, speciesIndex++);
            }
            data[4] = speciesMap.get(species);
            processedData.add(data);
        }
        return processedData;
    }
}