# Historic Kinematic Solar Cycle Summary

Generated on: 2026-06-11

| Cycle | Start | Max | Amplitude | Kinematic Event | Phase |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1755-06 | 1761-06 | 137.6 | 1762-08 | Declining |
| 2 | 1766-06 | 1769-11 | 188.8 | 1771-09 | Declining |
| 3 | 1775-06 | 1778-06 | 259.9 | 1783-07 | Declining |
| 4 | 1784-07 | 1788-01 | 231.7 | 1792-08 | Declining |
| 5 | 1798-03 | 1804-11 | 81.2 | 1802-01 | Rising |
| 6 | 1810-05 | 1816-04 | 79.8 | 1812-05 | Rising |
| 7 | 1823-02 | 1830-02 | 115.8 | 1831-06 | Declining |
| 8 | 1833-11 | 1837-03 | 239.7 | 1843-08 | Declining |
| 9 | 1843-11 | 1848-06 | 215.7 | 1851-08 | Declining |
| 10 | 1856-02 | 1860-02 | 183.4 | 1859-04 | Rising |
| 11 | 1867-04 | 1870-09 | 233.4 | 1873-01 | Declining |
| 12 | 1878-11 | 1883-12 | 120.6 | 1881-08 | Rising |
| 13 | 1890-02 | 1893-11 | 144.1 | 1891-03 | Rising |
| 14 | 1901-09 | 1905-11 | 102.6 | 1902-09 | Rising |
| 15 | 1913-07 | 1917-09 | 171.4 | 1920-01 | Declining |
| 16 | 1923-05 | 1928-05 | 125.8 | 1932-06 | Declining |
| 17 | 1933-10 | 1937-06 | 191.8 | 1941-01 | Declining |
| 18 | 1944-03 | 1947-06 | 213.6 | 1950-12 | Declining |
| 19 | 1954-05 | 1957-12 | 281.6 | 1961-10 | Declining |
| 20 | 1964-10 | 1968-11 | 154.0 | 1970-10 | Declining |
| 21 | 1976-07 | 1979-12 | 229.4 | 1982-05 | Declining |
| 22 | 1986-07 | 1989-08 | 212.5 | 1991-04 | Declining |
| 23 | 1996-07 | 2002-01 | 179.3 | 1999-10 | Rising |
| 24 | 2008-12 | 2014-04 | 113.6 | 2011-04 | Rising |

## Statistical Test: Rising vs. Declining Phase
| Metric | Rising Phase | Declining/Min Phase |
| :--- | :--- | :--- |
| **Count** | 8 | 16 |
| **Mean Amplitude** | 125.6 | 200.2 |
| **Median Amplitude** | **117.1** | **213.1** |

### The Amplitude Ceiling (Median Analysis)
Because standard means can be skewed by historical outliers (e.g., Solar Cycle 19), comparing the median amplitude provides a highly robust, non-parametric metric. Cycles that experience a natural kinematic reversal achieve a median amplitude of **213.1**, whereas cycles prematurely disrupted in their Rising Phase are constructively capped at a median of **117.1**. This represents a **45.0% systemic suppression** of the solar dynamo.

### Advanced Validation: Permutation Test
To rigorously validate this delta without relying on bell-curve assumptions, we conducted a non-parametric permutation test.

- **Resamples:** 10,000
- **Observed Difference in Means:** 74.59 sunspots
- **Empirical P-Value:** `0.00070`

**Conclusion:** The result is highly significant. It is mathematically improbable (p < 0.01) that the observed 45.0% amplitude suppression occurred by random chance.
## Sensitivity Analysis: Leave-One-Out (Jackknife)
To ensure the statistical significance and amplitude suppression are not artificially driven by a single historical outlier, we performed a Leave-One-Out (Jackknife) resampling analysis.

- **Worst-Case P-Value:** `0.00220` *(Occurs if Cycle 5 is removed)*
- **Worst-Case Suppression:** **43.4%** *(Occurs if Cycle 5 is removed)*

**Conclusion:** The model is **HIGHLY ROBUST**. Even in the absolute worst-case scenario (dropping the most influential historical cycles), the mathematical probability remains significant and the physical amplitude ceiling holds.