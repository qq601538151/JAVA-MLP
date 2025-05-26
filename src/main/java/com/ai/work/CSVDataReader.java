package com.ai.work;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class CSVDataReader {
    public static List<String[]> readCSV(String filePath) {
        List<String[]> data = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new InputStreamReader(
                CSVDataReader.class.getClassLoader().getResourceAsStream(filePath)))) {
            if (reader != null) {
                data = reader.readAll();
            }
        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
        return data;
    }
}