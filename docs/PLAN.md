# HingRoute: Phase 1 Plan

## 1. Selected Intents
- `lost_or_stolen_card` (Risk)
- `compromised_card` (Risk)
- `card_not_working`
- `card_arrival`
- `transfer_not_received_by_recipient`
- `cancel_transfer`
- `transfer_fee_charged`
- `pending_card_payment`
- `declined_card_payment`
- `extra_charge_on_statement` (Risk)
- `top_up_failed`
- `other` (10% sampled from remaining Banking77 intents)

## 2. Hard Risk Overrides
The following intents bypass the confidence threshold and escalate to human review immediately:
- `lost_or_stolen_card`
- `compromised_card`
- `extra_charge_on_statement`
- `other`

## 3. Evaluation Metrics
- Primary: Macro-F1 across all 12 classes with 95% bootstrap confidence interval.
- Operating metrics: Auto-handled coverage (share above tau) vs accuracy on auto-handled queries.
- Safety metric: 100% recall on risk intents reaching escalation.

## 4. Models to Compare
1. Baseline 0: Majority Class Predictor
2. Baseline 1: Character n-gram TF-IDF + Logistic Regression
3. Baseline 2: Zero-shot LLM Prompt
4. Fine-tuned: XLM-RoBERTa-base
5. Fine-tuned: MuRIL-base-cased