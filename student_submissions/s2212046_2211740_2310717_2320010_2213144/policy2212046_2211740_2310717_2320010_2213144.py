from policy import Policy
import numpy as np
import random


class Policy2212046_2211740_2310717_2320010_2213144(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = BestFitDecrease()
        elif policy_id == 2:
            self.policy = RandomPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


class BestFitDecrease(Policy):
    def __init__(self):
        self.sorted_prods = None
        self.sorted_stocks = None
        self.stock_area_used = None
        self.can_place_prod = None

    def get_action(self, observation, info):
        if info.get("filled_ratio", 0.0) == 0.0:
            # diện tích trung bình và số lượng trung bình
            prod_average_area = sum(product["size"][0] * product["size"][1] * product["quantity"] for product in
                                    observation["products"]) / len(observation["products"])
            prod_average_quantity = sum(product["quantity"] for product in observation["products"]) / len(
                observation["products"])

            # diện tích giảm dần
            self.sorted_stocks = sorted(
                enumerate(observation["stocks"]),
                key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
                reverse=True
            )

            self.sorted_prods = sorted(
                [prod for prod in observation["products"] if prod["quantity"] > 0],
                key=lambda x: x["size"][0] * x["size"][1],
                reverse=True
            )

            # nếu diện tích trung bình * số lượng trung bình < diện tích của tấm stock nhỏ nhất, đảo ngược thứ tự tấm stock
            if len(self.sorted_stocks) >= 100 and prod_average_area * prod_average_quantity < \
                    self._get_stock_size_(self.sorted_stocks[99][1])[0] * \
                    self._get_stock_size_(self.sorted_stocks[99][1])[1]:
                self.sorted_stocks = self.sorted_stocks[::-1]

            self.stock_area_used = np.zeros(len(self.sorted_stocks))
            self.can_place_prod = np.ones((len(self.sorted_stocks), len(self.sorted_prods)), dtype=bool)

        for prod_idx, prod in enumerate(self.sorted_prods):
            if prod["quantity"] == 0:
                continue
            prod_w, prod_h = prod["size"]
            for stock_idx, stock in self.sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                area_not_sufficient = self.stock_area_used[stock_idx] + prod_w * prod_h > stock_w * stock_h
                if area_not_sufficient or not self.can_place_prod[stock_idx, prod_idx]:
                    continue

                # Xoay
                orientations = [
                    (prod_w, prod_h),
                    (prod_h, prod_w) if prod_w != prod_h else None
                ]
                for orientation in orientations:
                    if orientation is None:
                        continue
                    current_w, current_h = orientation
                    if stock_w >= current_w and stock_h >= current_h:
                        for x in range(stock_w - current_w + 1):
                            for y in range(stock_h - current_h + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    x_pos, y_pos = x, y
                                    self.stock_area_used[stock_idx] += current_w * current_h
                                    # Đánh dấu sản phẩm đã được đặt
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": list(orientation),
                                        "position": (x_pos, y_pos)
                                    }
                # Khong the dat
                self.can_place_prod[stock_idx, prod_idx] = False

        # trả về mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_stock_size_(self, stock):
        # đếm số hàng và cột có ít nhất một ô khác -2
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        # kiểm tra xem có thể đặt không
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        # kiểm tra vùng cần đặt sản phẩm có trống không
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # chọn ngẫu nhiên một sản phẩm ( > 0)
        available_prods = [prod for prod in list_prods if prod["quantity"] > 0]
        if not available_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        prod = random.choice(available_prods)
        prod_size = prod["size"]
        prod_w, prod_h = prod_size

        # chọn ngẫu nhiên một tấm stock
        for _ in range(100):
            stock_idx = random.randint(0, len(stocks) - 1)
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            # Check xem co xoay duoc khong
            orientations = [
                (prod_w, prod_h),
                (prod_h, prod_w) if prod_w != prod_h else None
            ]
            orientations = [o for o in orientations if o is not None]
            if not orientations:
                continue
            orientation = random.choice(orientations)
            current_w, current_h = orientation
            if stock_w >= current_w and stock_h >= current_h:
                # chon vi tri ngau nhien
                pos_x = random.randint(0, stock_w - current_w)
                pos_y = random.randint(0, stock_h - current_h)
                if self._can_place_(stock, (pos_x, pos_y), orientation):
                    return {
                        "stock_idx": stock_idx,
                        "size": list(orientation),
                        "position": (pos_x, pos_y)
                    }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)
