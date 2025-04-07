function applyFilters() {
  const symbolFilter = document.getElementById("symbolFilter").value.toUpperCase();
  const priceFilter = parseFloat(document.getElementById("priceFilter").value);
  const marketCapFilter = parseFloat(document.getElementById("marketCapFilter").value);

  let url = "/api/stocks/filter?";
  if (symbolFilter) url += `symbol=${symbolFilter}&`;
  if (!isNaN(priceFilter)) url += `price=${priceFilter}&`;
  if (!isNaN(marketCapFilter)) url += `marketCap=${marketCapFilter}`;

  fetch(url)
      .then(response => response.json())
      .then(data => {
          displayStocks(data);
          updateCharts(data);
      });
}
