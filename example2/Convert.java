import java.io.*;
import java.nio.file.*;
import java.util.*;

/*
 * Cleaning and Converting Original CSV into a Ready to Train Dataset for Breans LinReg
 */

public class Convert {

    public static void main(String[] args) throws IOException {
        String inputFile = "nvdstock.csv";
        String outputFile = "nvdstock_numeric.csv";

        List<String> lines = Files.readAllLines(Paths.get(inputFile));
        if (lines.isEmpty()) {
            System.err.println("Input file is empty.");
            return;
        }

        // Skip header (first line)
        List<String> outputLines = new ArrayList<>();

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;

            // Split CSV by comma
            String[] parts = line.split(",");

            // Expected columns: 9 (Date, Open, Range, Volume, Log_Volume, Return_Percentage, 3_Day_Avg_AdjClose, PriorDay_AdjClose, Adj Close)
            if (parts.length < 9) {
                System.err.println("Skipping invalid line " + (i+1));
                continue;
            }

            // Extract numeric columns, skip Date (parts[0])
            // Convert Return_Percentage (parts[5]) from string with % to decimal
            String open = parts[1];
            String range = parts[2];
            String volume = parts[3];
            String logVolume = parts[4];
            String returnPercentStr = parts[5].replace("%", "");
            double returnPercent = 0.0;
            try {
                returnPercent = Double.parseDouble(returnPercentStr) / 100.0;
            } catch (NumberFormatException e) {
                System.err.println("Invalid Return_Percentage at line " + (i+1) + ": " + parts[5]);
                continue;
            }
            String avgAdjClose = parts[6];
            String priorDayAdjClose = parts[7];
            String adjClose = parts[8];

            // Build cleaned CSV line with columns: Open, Range, Volume, Log_Volume, Return_Percentage(decimal), 3_Day_Avg_AdjClose, PriorDay_AdjClose, Adj Close
            String outLine = String.join(",",
                open,
                range,
                volume,
                logVolume,
                String.valueOf(returnPercent),
                avgAdjClose,
                priorDayAdjClose,
                adjClose
            );

            outputLines.add(outLine);
        }

        // Write cleaned numeric CSV file
        Files.write(Paths.get(outputFile), outputLines);

        System.out.println("Converted file saved as: " + outputFile);
    }
}

