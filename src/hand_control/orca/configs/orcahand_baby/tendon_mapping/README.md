# Tendon mapping of the baby hand

The baby hand is the ORCA v2 scaled down, with its own tendon routing. This folder keeps the team's
tendon mapping (Notion page "Tendon Mapping", exported 23 Nov 2026):

| File | Content |
|---|---|
| `tendon_mapping.csv` | The mapping table: one row per tendon. `motor_check.py` reads it. |
| `tendon_mapping_table.pdf` | The original table as exported. |
| `tendon_mapping_notion_with_hole_pictures.pdf` | The Notion page with the pictures of the tendon holes at the wrist, for routing during assembly (one of its pictures shows an older thumb layout). |

Tendons are numbered in the order they cross the wrist between the metal pins (ORCA convention). Every motor
carries one antagonistic pair: one tendon on its top spool, its mate on the bottom spool. Codes: T/I/M/R/P = thumb,
index, middle, ring, pinky; AP = abduction (sideways), PP = proximal phalanx (knuckle, `_mcp`), DP = distal phalanx
(`_pip`, the thumb's `_dip`), TP = the thumb base (`thumb_cmc`).

## Motor -> joint

| Motor | Code | Joint (config.yaml) | Map entry |
|---|---|---|---|
| 1 | wrist | `wrist` | -1 |
| 2 | I-DP | `index_pip` | 2 |
| 3 | I-PP | `index_mcp` | -3 |
| 4 | T-AP | `thumb_abd` | 4 |
| 5 | T-DP | `thumb_dip` | 5 |
| 6 | M-AP | `middle_abd` | -6 |
| 7 | R-AP | `ring_abd` | 7 |
| 8 | R-DP | `ring_pip` | 8 |
| 9 | R-PP | `ring_mcp` | -9 |
| 10 | P-PP | `pinky_mcp` | 10 |
| 11 | P-DP | `pinky_pip` | -11 |
| 12 | P-AP | `pinky_abd` | -12 |
| 13 | M-PP | `middle_mcp` | -13 |
| 14 | M-DP | `middle_pip` | -14 |
| 15 | I-AP | `index_abd` | 15 |
| 16 | T-PP | `thumb_mcp` | -16 |
| 17 | T-TP | `thumb_cmc` | 17 |

The sign of a map entry is the joint's direction for positive motor rotation. It was found on the hand with
`motor_check.py` and in calibration (23 and 27 Nov 2026), not from the table; `config.yaml` is the reference and
notes which signs are still unconfirmed. Rule of thumb from the spools: a bending joint whose flex tendon is on the
top spool has a negative sign; exception: `thumb_dip`. If the routing changes, update `tendon_mapping.csv`, then
check the signs again with `motor_check.py --hand baby`.
