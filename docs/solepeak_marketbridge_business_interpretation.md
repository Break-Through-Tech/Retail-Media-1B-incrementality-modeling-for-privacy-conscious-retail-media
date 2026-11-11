# Business Interpretation: SolePeak × MarketBridge Campaign Incrementality

### Executive Takeaway
Targeting pays off, but broader reach is a waste. SolePeak should concentrate its media spend strictly on the top 10%–20% highest-uplift shoppers identified by causal machine learning. Broad targeting dilutes campaign effectiveness toward near-random performance, while unguided reach wastes budget on organic shoppers who would have visited anyway.

---

### What the Results Mean for SolePeak & MarketBridge

* **Recommendation: Focus Strictly on the Top 10%–20% of Shoppers**  
  The data tells a simple story: go narrow, not broad. In the top decile (the top 10% of users ranked by our models), our causal models drove roughly a 6% lift, creating around 900 truly net-new visits per 15,000 ad opportunities. Once you push beyond the top buckets, that extra lift drops off hard and starts looking like random guessing (~1.7%). For SolePeak, spending ad budget outside the top tiers is essentially throwing money away. For MarketBridge, selling these top-tier shoppers as a specialized, proven high-impact segment justifies a premium price.  
  *Takeaway: Spend only on the top deciles where ads generate genuine visit spikes rather than wasted impressions.*

* **Why This Measures "Trying to Show an Ad," Not "Actually Staring at It" (ITT)**  
  Our models look at what happens when the ad system *decides to target someone* (treatment), not whether the shopper actually looked at the banner on their screen (exposure). In the real world, an ad can win an auction but end up below the fold or ignored as someone quickly scrolls past. If we only studied people who 100% saw the ad, we would bias our numbers by selecting for slower or more engaged shoppers. Measuring the decision to target gives SolePeak the real, honest ROI of their bidding strategy under realistic production conditions.  
  *Takeaway: Evaluating targeting decisions instead of physical ad views gives an honest, real-world estimate of campaign ROI without selection bias.*

* **Predicting Visits vs. Causing Visits (The "Always-Buyer" Trap)**  
  Regular machine learning looks for people most likely to visit. Sounds fine until you realize who that is: people who already love the brand. Our standard response model targeted people with a massive 30.5% baseline visit rate without ever seeing an ad. That isn't great marketing—it's paying to advertise to loyal shoppers who were opening the app anyway. Causal uplift models flip the goal: they ignore the "sure things" and find the persuadable shoppers who showed up *only because* an ad nudged them.  
  *Takeaway: Standard models waste ad dollars on people already buying; causal models target only those who visit because of the ad.*

* **A Quick Look at Conversions**  
  Our models were trained to drive incremental visits, not purchases directly. But looking at the downstream checkout numbers descriptively, the people in our top visit tiers also showed strong checkout activity. That confirms our models aren't just attracting random window-shoppers—they're driving real, qualified interest that carries all the way through to buying.  
  *Takeaway: Optimizing for incremental site visits successfully brings in high-intent shoppers who complete purchases downstream.*