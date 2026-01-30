import os
import stripe

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

def create_checkout_session(price_id: str, success_url: str, cancel_url: str):
    if not stripe.api_key:
        raise ValueError("STRIPE_SECRET_KEY is not set.")
    return stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
    )
