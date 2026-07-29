import os
import json
import pandas as pd
import instructor
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

class MarketRationale(BaseModel):
    order_book_analysis: str = Field(description="Reasoning about bid/ask imbalance.")
    macro_sentiment_alignment: str = Field(description="Market momentum context.")

class TradingAction(BaseModel):
    action_type: Literal["BUY", "SELL", "HOLD", "CANCEL"]
    price: float
    size: int

    @model_validator(mode='after')
    def validate_logic(self):
        if self.action_type in ["HOLD", "CANCEL"] and (self.price != 0.0 or self.size != 0):
            raise ValueError("HOLD/CANCEL must have 0 price and size.")
        return self

class AgentDecision(BaseModel):
    rationale: MarketRationale
    action: TradingAction

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    ),
    mode=instructor.Mode.TOOLS
)

def download_kaggle_dataset(dataset_slug: str, target_file: str) -> str:
    """
    Programmatically authenticates and downloads a dataset from Kaggle 
    using credentials provided securely through the .env file.
    """

    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    print("Authenticating with Kaggle API via environment variables...")
    api.authenticate()
    
    if os.path.exists(target_file):
        print(f"Dataset asset '{target_file}' already exists locally. Skipping download.")
        return target_file
        
    print(f"Downloading dataset '{dataset_slug}' from Kaggle...")
    api.dataset_download_files(dataset_slug, path=".", unzip=True)
    print("Download and extraction complete.")
    return target_file

class MarketBookSimulator:
    def __init__(self):
        self.bids = pd.DataFrame(columns=["price", "size"])
        self.asks = pd.DataFrame(columns=["price", "size"])

    def update_book(self, side: str, price: float, size: float):
        target_df = self.bids if side == "buy" else self.asks
        if size == 0:
            target_df = target_df[target_df["price"] != price]
        else:
            if price in target_df["price"].values:
                target_df.loc[target_df["price"] == price, "size"] = size
            else:
                new_row = pd.DataFrame([{"price": price, "size": size}])
                target_df = pd.concat([target_df, new_row], ignore_index=True)

        if side == "buy":
            self.bids = target_df.sort_values(by="price", ascending=False).reset_index(drop=True)
        else:
            self.asks = target_df.sort_values(by="price", ascending=True).reset_index(drop=True)

    def generate_llm_log(self, ticker: str, timestamp: str, depth: int = 5) -> str:
        log_lines = [f"=== MARKET STATE START ===\nTimestamp: {timestamp} | Ticker: {ticker}"]
        log_lines.append("[ASK QUEUE]")
        for _, row in self.asks.head(depth).iloc[::-1].iterrows():
            log_lines.append(f"  Ask: {row['price']:.2f} | Size: {int(row['size'])}")
        log_lines.append("[BID QUEUE]")
        for _, row in self.bids.head(depth).iterrows():
            log_lines.append(f"  Bid: {row['price']:.2f} | Size: {int(row['size'])}")
        log_lines.append("=== MARKET STATE END ===")
        return "\n".join(log_lines)

def run_rollout_collection(dataset_slug: str, csv_filename: str, output_jsonl_path: str):
    sim = MarketBookSimulator()
    
    csv_file_path = download_kaggle_dataset(dataset_slug, csv_filename)
    
    historical_ticks = pd.read_csv(csv_file_path)
    print(f"Starting Rollout... Processing {len(historical_ticks)} ticks.")

    with open(output_jsonl_path, "a") as dataset_file:
        for index, tick in historical_ticks.iterrows():
            
            mapped_side = "buy" if tick["side"] == "bid" else "sell"
            
            sim.update_book(
                side=mapped_side, 
                price=float(tick["price"]), 
                size=float(tick["quantity"])
            )
            
            state_prompt = sim.generate_llm_log("BTC-USDT", tick["timestamp"], depth=3)
            
            try:
                decision = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=AgentDecision,
                    messages=[
                        {"role": "system", "content": "You are an expert high-frequency trading algorithm."},
                        {"role": "user", "content": state_prompt}
                    ]
                )
                
                training_example = {
                    "messages": [
                        {"role": "system", "content": "You are an expert high-frequency trading algorithm."},
                        {"role": "user", "content": state_prompt},
                        {"role": "assistant", "content": decision.model_dump_json()}
                    ]
                }
                
                dataset_file.write(json.dumps(training_example) + "\n")
                print(f"Processed tick {index+1}/{len(historical_ticks)} successfully.")
                
            except Exception as e:
                print(f"Failed on tick {index+1}: {e}")

if __name__ == "__main__":
    run_rollout_collection(
        dataset_slug="fast42/btc-l2-order-book-btcusdt-1s-11825", 
        csv_filename="2025-08-11.csv", 
        output_jsonl_path="student_training_dataset.jsonl"
    )