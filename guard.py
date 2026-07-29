import os
import instructor
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from typing import Literal


class MarketRationale(BaseModel):
    order_book_analysis: str = Field(
        description="Detailed reasoning about bid/ask imbalance and liquidity walls."
    )
    macro_sentiment_alignment: str = Field(
        description="Explanation of how the decision aligns with current market momentum."
    )

class TradingAction(BaseModel):
    action_type: Literal["BUY", "SELL", "HOLD", "CANCEL"]
    
    price: float = Field(
        description="Execution price. Must be 0.0 if action is HOLD or CANCEL."
    )
    size: int = Field(
        description="Number of shares to route. Must be 0 if action is HOLD or CANCEL."
    )

    @model_validator(mode='after')
    def validate_hold_cancel_logic(self):
        if self.action_type in ["HOLD", "CANCEL"]:
            if self.price != 0.0 or self.size != 0:
                raise ValueError(f"If action is {self.action_type}, price and size must be 0.")
        elif self.action_type in ["BUY", "SELL"]:
            if self.price <= 0 or self.size <= 0:
                raise ValueError(f"If action is {self.action_type}, price and size must be greater than 0.")
        return self

class AgentDecision(BaseModel):
    """The master schema that the LLM MUST return."""
    rationale: MarketRationale
    action: TradingAction

client = instructor.from_openai(OpenAI())

def get_trading_decision(market_state_log: str) -> AgentDecision:
    """
    Passes the textual market state to the Teacher LLM and forces it to 
    return a validated AgentDecision Pydantic object.
    """
    try:
        decision = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=AgentDecision,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": "You are a high-frequency trading agent. Analyze the provided Level 2 order book state and output your strategic decision."
                },
                {
                    "role": "user",
                    "content": market_state_log
                }
            ]
        )
        return decision
    
    except Exception as e:
        print(f"Schema Guard failed to parse LLM output: {e}")
        return AgentDecision(
            rationale=MarketRationale(
                order_book_analysis="ERROR OR TIMEOUT", 
                macro_sentiment_alignment="ERROR OR TIMEOUT"
            ),
            action=TradingAction(action_type="HOLD", price=0.0, size=0)
        )

if __name__ == "__main__":
    sample_market_log = """
    === MARKET STATE START ===
    Timestamp: 09:30:00.600 | Ticker: AAPL
    Best Bid: 150.00 | Best Ask: 150.02 | Spread: 0.02
    Order Book Imbalance (Bid/Total Depth): 0.93

    [ASK QUEUE (Sellers)]
      Ask Price: 150.10 | Size: 800
      Ask Price: 150.02 | Size: 150
    [BID QUEUE (Buyers)]
      Bid Price: 150.00 | Size: 750
      Bid Price: 149.95 | Size: 1200
    === MARKET STATE END ===
    """

    print("Querying Teacher LLM through the Schema Guard...\n")
    
    validated_decision = get_trading_decision(sample_market_log)
    
    print("--- VALIDATED OBJECT PROPERTIES ---")
    print(f"Action Type:  {validated_decision.action.action_type}")
    print(f"Trade Size:   {validated_decision.action.size}")
    print(f"Trade Price:  {validated_decision.action.price}")
    print(f"\nOrder Book Rationale: {validated_decision.rationale.order_book_analysis}")
    
    # Because it is a Pydantic model, you can easily dump it back to a clean JSON dictionary for your dataset
    # print(validated_decision.model_dump_json(indent=2))