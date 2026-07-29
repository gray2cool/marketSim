import json
from datetime import datetime
import pandas as pd


class MarketBookSimulator:

    def __init__(self):
        # Dataframes to track current Level 2 Order Book State
        self.bids = pd.DataFrame(columns=["price", "size"])
        self.asks = pd.DataFrame(columns=["price", "size"])
        self.log_entries = []

    def update_book(self, side: str, price: float, size: float):
        """Updates the internal L2 book state by modifying or removing price levels."""
        target_df = self.bids if side == "buy" else self.asks

        if size == 0:
            # Remove price level if size is zero
            target_df = target_df[target_df["price"] != price]
        else:
            # If price level exists, update it; otherwise, append it
            if price in target_df["price"].values:
                target_df.loc[target_df["price"] == price, "size"] = size
            else:
                new_row = pd.DataFrame([{"price": price, "size": size}])
                target_df = pd.concat([target_df, new_row], ignore_index=True)

        # Sort Bids descending (highest first) and Asks ascending (lowest first)
        if side == "buy":
            self.bids = target_df.sort_values(
                by="price", ascending=False
            ).reset_index(drop=True)
        else:
            self.asks = target_df.sort_values(
                by="price", ascending=True
            ).reset_index(drop=True)

    def generate_llm_log(
        self, ticker: str, timestamp: str, depth: int = 5
    ) -> str:
        """Converts the current state of the order book into a structured token-friendly text block."""
        # Extract top N levels
        top_bids = self.bids.head(depth)
        top_asks = self.asks.head(depth)

        # Calculate basic order book metrics
        best_bid = top_bids.iloc[0]["price"] if not top_bids.empty else None
        best_ask = top_asks.iloc[0]["price"] if not top_asks.empty else None
        spread = best_ask - best_bid if (best_bid and best_ask) else 0.0

        total_bid_depth = top_bids["size"].sum()
        total_ask_depth = top_asks["size"].sum()
        imbalance = (
            total_bid_depth / (total_bid_depth + total_ask_depth)
            if (total_bid_depth + total_ask_depth) > 0
            else 0.5
        )

        # Format the text prompt/log
        log_lines = [
            f"=== MARKET STATE START ===",
            f"Timestamp: {timestamp} | Ticker: {ticker}",
            f"Best Bid: {best_bid} | Best Ask: {best_ask} | Spread: {spread:.2f}",
            f"Order Book Imbalance (Bid/Total Depth): {imbalance:.2f}",
            "\n[ASK QUEUE (Sellers)]",
        ]

        # Add asks in reverse order so the lowest ask sits right above the bid
        for _, row in top_asks.iloc[::-1].iterrows():
            log_lines.append(f"  Ask Price: {row['price']:.2f} | Size: {int(row['size'])}")

        log_lines.append("[BID QUEUE (Buyers)]")
        for _, row in top_bids.iterrows():
            log_lines.append(f"  Bid Price: {row['price']:.2f} | Size: {int(row['size'])}")

        log_lines.extend(
            [
                "\n[EXPECTED TEACHER OUTPUT SCHEMA]",
                "Your output must follow this JSON format exactly:",
                json.dumps(
                    {
                        "Rationale": {
                            "OrderBookAnalysis": "Reasoning about imbalance and liquidity wall shifts...",
                            "MacroSentimentAlignment": "How this matches broader market trend...",
                        },
                        "Action": {
                            "Type": "BUY | SELL | HOLD | CANCEL",
                            "Price": 0.00,
                            "Size": 0,
                        },
                    },
                    indent=2,
                ),
                "=== MARKET STATE END ===",
            ]
        )

        return "\n".join(log_lines)


# ==========================================
# SIMULATION EXECUTION & DATA SEEDING
# ==========================================
if __name__ == "__main__":
    sim = MarketBookSimulator()

    # Generate synthetic historical raw order flow data stream
    raw_market_events = [
        {"time": "09:30:00.000", "side": "buy", "price": 150.00, "size": 500},
        {"time": "09:30:00.100", "side": "buy", "price": 149.95, "size": 1200},
        {"time": "09:30:00.200", "side": "sell", "price": 150.05, "size": 300},
        {"time": "09:30:00.300", "side": "sell", "price": 150.10, "size": 800},
        {"time": "09:30:00.400", "side": "buy", "price": 150.00, "size": 750},  # Size increase
        {"time": "09:30:00.500", "side": "sell", "price": 150.02, "size": 150},  # Price improvement
        {"time": "09:30:00.600", "side": "sell", "price": 150.05, "size": 0},  # Cancellation
    ]

    print("Streaming events into the simulator and exporting logs...\n")

    # Process events sequentially through our event-driven state container
    for event in raw_market_events:
        sim.update_book(
            side=event["side"], price=event["price"], size=event["size"]
        )

        # Generate state log entry for each update tick
        state_log = sim.generate_llm_log(
            ticker="AAPL", timestamp=event["time"], depth=3
        )
        sim.log_entries.append(state_log)

    # Output the final state log snapshot to demonstrate the structure
    print(sim.log_entries[-1])

    # Save the log entries dataset to a local text file for later training token evaluation
    with open("market_simulator_prompt_logs.txt", "w") as f:
        f.write("\n\n#########################################\n\n".join(sim.log_entries))
    
    print("\nLogs successfully exported to 'market_simulator_prompt_logs.txt'")