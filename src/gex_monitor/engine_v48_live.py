"""KDJ v4.8b indicator classes for live trading."""

from collections import deque


class KDJCalculator:
    def __init__(self, period=9, k_smooth=3, d_smooth=3):
        self.period = period
        self.highs = deque(maxlen=period)
        self.lows = deque(maxlen=period)
        self.rsv_buf = deque(maxlen=k_smooth)
        self.k_buf = deque(maxlen=d_smooth)
        self.k = self.d = self.j = 50.0
        self.prev_j = self.prev_prev_j = 50.0
        self.ready = False
        self._count = 0
        self._warmup = period + k_smooth + d_smooth

    def update(self, high, low, close):
        self.prev_prev_j = self.prev_j
        self.prev_j = self.j
        self.highs.append(float(high))
        self.lows.append(float(low))
        self._count += 1
        if self._count < self.period:
            return self.k, self.d, self.j
        hh, ll = max(self.highs), min(self.lows)
        rsv = (float(close) - ll) / (hh - ll) * 100 if hh != ll else 50.0
        self.rsv_buf.append(rsv)
        self.k = sum(self.rsv_buf) / len(self.rsv_buf)
        self.k_buf.append(self.k)
        self.d = sum(self.k_buf) / len(self.k_buf)
        self.j = 3 * self.k - 2 * self.d
        if self._count >= self._warmup:
            self.ready = True
        return self.k, self.d, self.j

    def simulate_j(self, hypothetical_close):
        """假设当前5min bar以此价格收盘，J会是多少？只读，不改内部状态。"""
        if self._count < self.period or len(self.highs) == 0:
            return self.j
        c = float(hypothetical_close)
        hh = max(max(self.highs), c)
        ll = min(min(self.lows), c)
        rsv = (c - ll) / (hh - ll) * 100 if hh != ll else 50.0
        sim_rsv = list(self.rsv_buf) + [rsv]
        if len(sim_rsv) > self.rsv_buf.maxlen:
            sim_rsv = sim_rsv[-self.rsv_buf.maxlen:]
        sim_k = sum(sim_rsv) / len(sim_rsv)
        sim_k_buf = list(self.k_buf) + [sim_k]
        if len(sim_k_buf) > self.k_buf.maxlen:
            sim_k_buf = sim_k_buf[-self.k_buf.maxlen:]
        sim_d = sum(sim_k_buf) / len(sim_k_buf)
        return 3 * sim_k - 2 * sim_d


class ADXCalculator:
    def __init__(self, period=14):
        self.period = period
        self._ph = self._pl = self._pc = None
        self._tr_s = self._pdm_s = self._mdm_s = 0.0
        self._adx = 0.0
        self._dx_sum = 0.0
        self._count = 0
        self.ready = False

    def update(self, high, low, close):
        h, l, c = float(high), float(low), float(close)
        if self._ph is None:
            self._ph, self._pl, self._pc = h, l, c
            return 0.0
        tr = max(h - l, abs(h - self._pc), abs(l - self._pc))
        up, dn = h - self._ph, self._pl - l
        pdm = up if up > dn and up > 0 else 0
        mdm = dn if dn > up and dn > 0 else 0
        self._count += 1
        n = self.period
        if self._count <= n:
            self._tr_s += tr
            self._pdm_s += pdm
            self._mdm_s += mdm
        else:
            self._tr_s = self._tr_s - self._tr_s / n + tr
            self._pdm_s = self._pdm_s - self._pdm_s / n + pdm
            self._mdm_s = self._mdm_s - self._mdm_s / n + mdm
        self._ph, self._pl, self._pc = h, l, c
        if self._count < n:
            return 0.0
        pdi = self._pdm_s / self._tr_s * 100 if self._tr_s > 0 else 0
        mdi = self._mdm_s / self._tr_s * 100 if self._tr_s > 0 else 0
        di_sum = pdi + mdi
        dx = abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0
        if self._count == n:
            self._adx = dx
            self._dx_sum = dx
        elif self._count <= n * 2:
            self._dx_sum += dx
            if self._count == n * 2:
                self._adx = self._dx_sum / n
                self.ready = True
        else:
            self._adx = (self._adx * (n - 1) + dx) / n
        return self._adx


class ATRCalculator:
    def __init__(self, period=14):
        self._buf = deque(maxlen=period)
        self._pc = None
        self.value = 0.0

    def update(self, high, low, close):
        h, l, c = float(high), float(low), float(close)
        tr = max(h - l, abs(h - self._pc), abs(l - self._pc)) if self._pc else h - l
        self._pc = c
        self._buf.append(tr)
        self.value = sum(self._buf) / len(self._buf)
        return self.value


class DEMACalculator:
    def __init__(self, period):
        self.alpha = 2.0 / (period + 1)
        self.ema1 = self.ema2 = None
        self.value = None
        self._n = 0
        self._warmup = period * 2

    def update(self, price):
        p = float(price)
        self._n += 1
        if self.ema1 is None:
            self.ema1 = self.ema2 = p
        else:
            self.ema1 = self.alpha * p + (1 - self.alpha) * self.ema1
            self.ema2 = self.alpha * self.ema1 + (1 - self.alpha) * self.ema2
        self.value = 2 * self.ema1 - self.ema2
        return self.value

    @property
    def ready(self):
        return self._n >= self._warmup


class SpeedDivergence:
    def __init__(self, div_len=5, div_thresh=1.5, j_bull_max=30, j_bear_min=70):
        self.div_len = div_len
        self.div_thresh = div_thresh
        self.j_bull_max = j_bull_max
        self.j_bear_min = j_bear_min
        self._closes = deque(maxlen=div_len + 1)
        self._highs50 = deque(maxlen=50)
        self._lows50 = deque(maxlen=50)
        self._j_hist = deque(maxlen=div_len + 1)
        self._prev_bull = self._prev_bear = False

    def update(self, high, low, close, j):
        self._closes.append(float(close))
        self._highs50.append(float(high))
        self._lows50.append(float(low))
        self._j_hist.append(float(j))
        if len(self._closes) <= self.div_len:
            return False, False
        price_range = max(self._highs50) - min(self._lows50)
        if price_range <= 0:
            self._prev_bull = self._prev_bear = False
            return False, False
        pc = (self._closes[-1] - self._closes[-1 - self.div_len]) / price_range * 100
        jc = self._j_hist[-1] - self._j_hist[-1 - self.div_len]
        ps, js = abs(pc), abs(jc)
        sr = js / ps if ps > 0.1 else 0.0
        raw_bull = pc < 0 and jc < 0 and sr > self.div_thresh and j < self.j_bull_max
        raw_bear = pc > 0 and jc > 0 and sr > self.div_thresh and j > self.j_bear_min
        bull_just = raw_bull and not self._prev_bull
        bear_just = raw_bear and not self._prev_bear
        self._prev_bull, self._prev_bear = raw_bull, raw_bear
        return bull_just, bear_just
