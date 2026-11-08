# Scenario data

Each scenario folder is a complete input dataset.

## Units

- Coordinates and road `distance`: metres.
- Road `speed`: metres per second (30 on main roads, 15 on secondary roads).
- All simulation times and durations: seconds, including incident timestamps, arrival deadlines, handling durations, responder availability, and assignment start times. Fractional seconds are allowed.

## Interpretation

- Road reports are available at time 0. Weather is local to the reported road.
- Traffic observations are historical and available before time 0. `observation_nr` orders a separate sequence for each directed road; it is not a simulation timestamp.
- Traffic levels (Low/Medium/High) differ from road statuses (Open/Congested/Blocked). Define and justify how traffic affects your planning; no fixed traffic multipliers are prescribed.

Document and defend any additional assumptions.
