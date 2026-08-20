# Data Dictionary

Each row is one anonymized record from a randomized advertising experiment. The source does not
provide a unique customer identifier, direct personal information, or the business meaning of its
anonymous features.

| Column | Expected type | Source meaning | Project use |
|---|---|---|---|
| `f0` to `f11` | Numeric | Twelve anonymized and transformed features | The only audience features permitted as model inputs. Do not invent business meanings for them. |
| `treatment` | Binary, 0 or 1 | Random treatment assignment. `1` is treatment and `0` is control. | Use to define treatment and control groups and to construct uplift methods. Do not include it in the normal audience feature list. |
| `conversion` | Binary, 0 or 1 | Whether a conversion occurred on an advertiser website | Optional secondary outcome because conversions are rare. |
| `visit` | Binary, 0 or 1 | Whether a visit occurred on an advertiser website | Primary project outcome. |
| `exposure` | Binary, 0 or 1 | Whether advertising exposure was observed | Diagnostic field only for the primary analysis. Do not use it as a normal audience feature because it occurs after treatment assignment. |

## Fictional Business Interpretation

The project uses SolePeak Footwear and MarketBridge Retail Media as fictional stakeholders. This
mapping is provided only to make the business problem easier to understand. It does not describe
the actual companies, data contributors, business context, or feature meanings in the Criteo data.

| Dataset columns | Project assumption |
|---|---|
| `f0` to `f11` | In this project, we assume these anonymous features are derived from approved audience and behavioral signals contributed by SolePeak, MarketBridge, or both companies. The actual source of each feature is unknown. |
| `treatment` | In this project, we assume MarketBridge's campaign system records whether a customer was eligible to receive SolePeak's advertisement. |
| `exposure` | In this project, we assume MarketBridge's advertising delivery system records whether the advertisement was shown. |
| `visit` | For the fictional scenario, we interpret this as an approved visit outcome available to MarketBridge. It should not be presented as the dataset's actual retail-property visit definition. |
| `conversion` | For the fictional scenario, we interpret this as an approved conversion outcome available to MarketBridge. It should not be presented as the dataset's actual footwear-purchase definition. |

Privacy-safe identifiers might be used inside a real data clean room to match approved records.
Those identifiers are not included in these files. The dataset is an anonymized analytical table,
not an identity graph or a production data clean room. Criteo does not state that the source data
was produced through a data clean room.

## Modeling Guardrails

- Use only `f0` through `f11` in the normal model feature matrix
- Use `treatment` to define the randomized comparison
- Keep `visit` as the primary outcome
- Treat `conversion` as an optional secondary outcome
- Exclude `exposure` from the primary audience feature matrix
- Do not filter the primary analysis to records where `exposure = 1`
- Fit preprocessing and models only on the training data
- Use validation data for model decisions and test data only for final evaluation
- Report treatment and outcome rates for each split
- Report conclusions using aggregate or cohort-level results
