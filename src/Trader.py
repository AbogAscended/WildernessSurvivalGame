"""Trader entity with difficulty-scaled bartering behavior.

The :class:`Trader` is implemented as a specialized repeating item. It accepts
offers in terms of gold, water, and food and may counter based on its leniency
(``deviation``) and a maximum number of counter offers. Profiles and sampling
are driven by :mod:`src.Difficulty`.
"""

import random

# Prefer package-relative imports; fall back to absolute when run directly
try:  # pragma: no cover - import resolution shim
    from . import Items, Difficulty  # type: ignore
except Exception:  # pragma: no cover
    import Items  # type: ignore
    try:
        import Difficulty  # type: ignore
    except Exception:
        Difficulty = None


class Trader(Items.Items):
    """A repeating item that represents a trader NPC.

    :param str difficulty: Difficulty label used to select a trader behavior
        profile. In absence of :mod:`Difficulty`, sensible defaults are used.
    :param random.Random rng: Optional RNG for reproducible proposal sampling.

    :ivar float deviation: Acceptable deviation around the original offer.
    :ivar int maxCounters: Maximum number of counter offers allowed.
    :ivar int totalCounters: Number of counters made so far in this session.
    :ivar list proposal: Current proposal ``[receiving, offering]`` with each
        side a triplet ``[gold, water, food]``.
    """

    deviation = maxCounters = totalCounters = 0
    proposal = None
    _rng = None

    def __init__(self,
                 difficulty,
                 tile_costs=(0, 0, 0),
                 rng: random.Random | None = None
                 ):
        super().__init__("trader", True, float("inf"))
        self._rng = rng or random.Random()
        self.tile_costs = tile_costs
        self.difficulty_str = difficulty

        # Resolve difficulty profile
        if Difficulty is not None:
            diff = Difficulty.canonicalize(difficulty)
            profile = Difficulty.get_trader_profile(diff)
            self.deviation = Difficulty.sample_deviation(self._rng, profile)
            self.maxCounters = profile["max_counters"].get(self.deviation, 2)
            self._total_range = profile.get("total_range", (2, 5))
        else:
            # Fallback defaults
            self.deviation = 1.0
            self.maxCounters = 3
            self._total_range = (2, 5)

        self.totalCounters = 0
        self.inventory = [self._generate_proposal() for _ in range(4)]

    def _generate_proposal(self):
        """Generate a proposal that scales with tile costs and difficulty."""
        # Calculate minimum viable amounts based on tile costs (move, water, food)
        m_cost, w_cost, f_cost = self.tile_costs
        
        # Cost to trade (wait action): roughly half the entry cost (ceil)
        w_trade = (w_cost + 1) // 2
        f_trade = (f_cost + 1) // 2
        
        # Total needed to break even (enter + trade)
        w_needed = w_cost + w_trade
        f_needed = f_cost + f_trade
        
        # Use the max resource cost as the baseline for significant trade value
        base_cost = max(w_needed, f_needed)
        # Ensure at least 1 to avoid zero-trades
        base_cost = max(1, base_cost)
        
        # 1. Determine Payment Amount (Keep relatively consistent with slight variance)
        payment_qty = self._rng.randint(base_cost, int(base_cost * 1.5))
        
        # 2. Determine Multiplier (Offer vs Payment Ratio) based on Difficulty
        d = self.difficulty_str.lower() if isinstance(self.difficulty_str, str) else "normal"
        
        if "easy" in d:
            # Easy: Skewed towards high values (Mean >> 1.0)
            mult = self._rng.triangular(1.2, 3.0, 2.8)
        elif "hard" in d:
            # Hard: Symmetric around profitable mean (e.g. 1.3)
            mult = self._rng.triangular(0.8, 1.8, 1.3)
        elif "extreme" in d:
            # Extreme: Skewed towards low/unprofitable values (Mode near Min)
            mult = self._rng.triangular(0.5, 1.5, 0.6)
        else: # Medium/Normal
            # Medium: Skewed high, but less extreme than Easy
            mult = self._rng.triangular(0.9, 2.2, 1.8)

        # 3. Calculate Offer Amount
        offer_qty = int(payment_qty * mult)
        # Ensure at least 1 (unless we really want 0? No, usually 1)
        offer_qty = max(1, offer_qty)
        
        # 1. Decide what Trader OFFERS (0=Gold, 1=Water, 2=Food)
        offer_type = self._rng.randint(0, 2)
        
        offering = [0, 0, 0]
        offering[offer_type] = offer_qty
        
        # 2. Decide what the Trader WANTS (what player pays)
        possible_wants = [0, 1, 2]
        if offer_type in possible_wants:
            possible_wants.remove(offer_type)
            
        want_type = self._rng.choice(possible_wants)
        
        wants = [0, 0, 0]
        wants[want_type] = payment_qty
        
        return [wants, offering]

    def getInventory(self):
        """Return the current list of proposals (inventory).

        :return: List of proposals, where each is ``[receiving, offering]``.
        """
        return self.inventory
