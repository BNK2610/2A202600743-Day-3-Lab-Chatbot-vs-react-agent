# Mock Data: Vietnam Trip Planner Agent

This folder contains CSV mock data for the Lab 3 domain:

```text
Vietnam Trip Planner Agent
```

The data is intentionally small, readable, and deterministic. It is designed for comparing a plain chatbot baseline with a ReAct agent that can call tools.

## Files

- `city_aliases.csv`: maps Vietnamese/user-friendly city names to normalized airport codes and city names.
- `trips.csv`: mock flight options with price, seats, and baggage fee.
- `weather.csv`: mock weather by city and date.
- `hotels.csv`: mock hotel availability and nightly prices.
- `activities.csv`: mock activity recommendations based on weather and city.
- `baggage_rules.csv`: included baggage and extra baggage fees by provider.
- `local_transport.csv`: estimated local transport cost from airport to city center.
- `test_scenarios.csv`: suggested evaluation scenarios for chatbot vs agent.
- `eval_ground_truth.csv`: checklist-style ground truth for scoring chatbot and agent outputs.

## Intentional Edge Cases

Use these cases for failure analysis and Agent v1 -> v2 improvement:

- No trip from `HAN` to `SGN` under `1000000` VND on `2026-06-10`.
- No route from `HAN` to `HUI` in `trips.csv`.
- `PQC` has only 1 seat left on the cheapest `HAN -> PQC` trip on `2026-06-12`.
- Test case `T11` is missing the travel date; the agent should ask a clarification question.
- Test case `T12` has no weather record for the requested city/date; the agent should not guess.
- Some scenarios require multiple tools: trip search, weather lookup, outfit recommendation, hotel search, transport estimate, and total cost calculation.

## Test Scenario Distribution

- Simple lookup: `T01`, `T06`, `T10`, `T13`
- Budget and missing-data edge cases: `T02`, `T08`, `T12`, `T14`, `T19`
- Multi-step planning: `T03`, `T15`, `T16`
- Full multi-tool planning: `T21`, `T22`
- Cost calculation: `T04`, `T17`
- Constraint handling: `T05`, `T11`, `T20`
- Activity and hotel planning: `T07`, `T09`, `T18`
