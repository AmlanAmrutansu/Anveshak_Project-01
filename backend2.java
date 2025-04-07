package com.stockhub.controller;

import com.stockhub.model.Stock;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/stocks")
public class StockController {

    private static List<Stock> stockData = new ArrayList<>();

    static {
        // Adding some stock data for demonstration
        stockData.add(new Stock("AAPL", "Apple Inc.", 172.14, 2850000000000L, 28.5, 35.1, 165.00, "0.55%", 182.94, 124.17, "Technology", 6.11, new double[]{150, 155, 160, 165, 170, 172}));
        stockData.add(new Stock("GOOGL", "Alphabet Inc.", 135.45, 1750000000000L, 23.3, 6.1, 140.00, "0%", 151.55, 83.45, "Communication Services", 5.76, new double[]{120, 125, 130, 132, 134, 135}));
        stockData.add(new Stock("MSFT", "Microsoft Corp.", 295.36, 2200000000000L, 34.2, 12.5, 310.00, "0.88%", 305.22, 212.03, "Technology", 8.05, new double[]{250, 260, 270, 280, 290, 295}));
        stockData.add(new Stock("AMZN", "Amazon.com Inc.", 125.12, 1270000000000L, 59.4, 8.9, 130.00, "0%", 145.86, 81.43, "Consumer Discretionary", 2.14, new double[]{100, 110, 115, 120, 123, 125}));
        stockData.add(new Stock("TSLA", "Tesla Inc.", 210.11, 670000000000L, 52.1, 14.2, 220.00, "0%", 293.34, 152.37, "Consumer Discretionary", 4.23, new double[]{180, 190, 200, 205, 208, 210}));
    }

    @GetMapping("/")
    public List<Stock> getStocks() {
        return stockData;
    }

    @GetMapping("/filter")
    public List<Stock> filterStocks(
            @RequestParam(value = "symbol", required = false) String symbol,
            @RequestParam(value = "price", required = false) Double price,
            @RequestParam(value = "marketCap", required = false) Double marketCap) {

        List<Stock> filteredStocks = new ArrayList<>();
        for (Stock stock : stockData) {
            boolean matchesSymbol = (symbol == null || stock.getSymbol().toLowerCase().contains(symbol.toLowerCase()));
            boolean matchesPrice = (price == null || stock.getPrice() <= price);
            boolean matchesMarketCap = (marketCap == null || stock.getMarketCap() >= marketCap);

            if (matchesSymbol && matchesPrice && matchesMarketCap) {
                filteredStocks.add(stock);
            }
        }
        return filteredStocks;
    }
}
