"""
紧急平仓脚本 — 平掉 TQQQ 所有持仓

用法:
  conda activate gex
  python close_tqqq.py              # 查看持仓
  python close_tqqq.py --execute    # 执行平仓
"""

import argparse
import time
from ib_insync import IB, Stock, MarketOrder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=4002)
    parser.add_argument('--client-id', type=int, default=50)
    parser.add_argument('--execute', action='store_true', help='实际执行平仓')
    args = parser.parse_args()

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
    print(f"Connected. Accounts: {ib.managedAccounts()}")

    # 查看所有持仓
    positions = ib.positions()
    tqqq_pos = [p for p in positions if p.contract.symbol == 'TQQQ']

    if not tqqq_pos:
        print("没有 TQQQ 持仓")
        ib.disconnect()
        return

    for p in tqqq_pos:
        qty = p.position
        cost = p.avgCost
        direction = "多" if qty > 0 else "空"
        print(f"  TQQQ {direction} {abs(qty):.0f}股  avgCost=${cost:.4f}  account={p.account}")

        if not args.execute:
            print(f"  → 加 --execute 执行平仓")
            continue

        # 平仓：反向下单
        contract = Stock('TQQQ', 'SMART', 'USD')
        ib.qualifyContracts(contract)
        side = 'SELL' if qty > 0 else 'BUY'
        close_qty = abs(int(qty))

        print(f"  → {side} {close_qty} TQQQ...")
        order = MarketOrder(side, close_qty)
        trade = ib.placeOrder(contract, order)

        start = time.time()
        while time.time() - start < 60:
            ib.sleep(0.5)
            if trade.orderStatus.status == 'Filled':
                fill = trade.orderStatus.avgFillPrice
                pnl = (fill - cost) * qty
                print(f"  ✓ FILLED @ ${fill:.2f}  PnL=${pnl:.2f}")
                break
        else:
            print(f"  ✗ 超时未成交，手动检查")

    ib.disconnect()
    print("Done.")

if __name__ == '__main__':
    main()
