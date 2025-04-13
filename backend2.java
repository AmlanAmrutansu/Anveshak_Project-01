/*
    A simplified example of a Java backend component.
    This example uses a minimal HTTP server functionality via com.sun.net.httpserver.HttpServer.
    In a production environment, you might use a framework like Spring Boot.
*/

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

public class backend2 {
    public static void main(String[] args) throws Exception {
        // Create an HTTP server on port 8000
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/api/stocks/filter", new StockFilterHandler());
        server.setExecutor(null); // creates a default executor
        System.out.println("Java backend running at http://localhost:8000/");
        server.start();
    }

    // Handler for processing stock filter requests.
    static class StockFilterHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("GET".equals(exchange.getRequestMethod())) {
                // Process query parameters
                String query = exchange.getRequestURI().getQuery(); // e.g. symbol=AAPL&price=150
                System.out.println("Received query: " + query);
                // Here, you could parse query parameters and use them to filter data.
                String jsonResponse = "{\"status\": \"success\", \"data\": [{\"symbol\": \"AAPL\", \"price\": 150.00}]}";
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, jsonResponse.getBytes(StandardCharsets.UTF_8).length);
                OutputStream os = exchange.getResponseBody();
                os.write(jsonResponse.getBytes(StandardCharsets.UTF_8));
                os.close();
            } else {
                String response = "{\"error\":\"Unsupported HTTP Method\"}";
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(405, response.getBytes(StandardCharsets.UTF_8).length);
                OutputStream os = exchange.getResponseBody();
                os.write(response.getBytes(StandardCharsets.UTF_8));
                os.close();
            }
        }
    }
}
