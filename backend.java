package com.stockhub.model;

public class Stock {
    private String symbol;
    private String name;
    private double price;
    private double marketCap;
    private double peRatio;
    private double pbRatio;
    private double intrinsicValue;
    private String dividendYield;
    private double high52Week;
    private double low52Week;
    private String sector;
    private double eps;
    private double[] history;  // Historical price trend data

    // Constructors, getters and setters

    public Stock(String symbol, String name, double price, double marketCap, double peRatio, double pbRatio, double intrinsicValue,
                 String dividendYield, double high52Week, double low52Week, String sector, double eps, double[] history) {
        this.symbol = symbol;
        this.name = name;
        this.price = price;
        this.marketCap = marketCap;
        this.peRatio = peRatio;
        this.pbRatio = pbRatio;
        this.intrinsicValue = intrinsicValue;
        this.dividendYield = dividendYield;
        this.high52Week = high52Week;
        this.low52Week = low52Week;
        this.sector = sector;
        this.eps = eps;
        this.history = history;
    }

    // Getters and setters
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public double getPrice() { return price; }
    public void setPrice(double price) { this.price = price; }
    public double getMarketCap() { return marketCap; }
    public void setMarketCap(double marketCap) { this.marketCap = marketCap; }
    public double getPeRatio() { return peRatio; }
    public void setPeRatio(double peRatio) { this.peRatio = peRatio; }
    public double getPbRatio() { return pbRatio; }
    public void setPbRatio(double pbRatio) { this.pbRatio = pbRatio; }
    public double getIntrinsicValue() { return intrinsicValue; }
    public void setIntrinsicValue(double intrinsicValue) { this.intrinsicValue = intrinsicValue; }
    public String getDividendYield() { return dividendYield; }
    public void setDividendYield(String dividendYield) { this.dividendYield = dividendYield; }
    public double getHigh52Week() { return high52Week; }
    public void setHigh52Week(double high52Week) { this.high52Week = high52Week; }
    public double getLow52Week() { return low52Week; }
    public void setLow52Week(double low52Week) { this.low52Week = low52Week; }
    public String getSector() { return sector; }
    public void setSector(String sector) { this.sector = sector; }
    public double getEps() { return eps; }
    public void setEps(double eps) { this.eps = eps; }
    public double[] getHistory() { return history; }
    public void setHistory(double[] history) { this.history = history; }
}
