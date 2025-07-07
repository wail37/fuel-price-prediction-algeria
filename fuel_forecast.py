import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. تحميل البيانات
df = pd.read_csv("fuel_prices_algeria.csv")

# 2. التأكد من صحة الأعمدة
df.columns = df.columns.str.strip()
print("📌 الأعمدة:", df.columns.tolist())

# 3. تحويل الأعمدة الرقمية
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Gasoline"] = pd.to_numeric(df["Gasoline"], errors="coerce")
df["Diesel"] = pd.to_numeric(df["Diesel"], errors="coerce")

# 4. تدريب النموذج
X = df[["Year"]]
y_gasoline = df["Gasoline"]
y_diesel = df["Diesel"]

model_gasoline = LinearRegression()
model_diesel = LinearRegression()

model_gasoline.fit(X, y_gasoline)
model_diesel.fit(X, y_diesel)

# 5. التنبؤ بالسعر للسنوات القادمة
future_years = pd.DataFrame({"Year": [2024, 2025]})
pred_gasoline = model_gasoline.predict(future_years)
pred_diesel = model_diesel.predict(future_years)

# 6. عرض النتائج
for year, g, d in zip(future_years["Year"], pred_gasoline, pred_diesel):
    print(f"📅 سنة {year}: بنزين ≈ {g:.2f} دج/لتر | ديزل ≈ {d:.2f} دج/لتر")

# 7. رسم بياني مع التوقعات
plt.plot(df["Year"], df["Gasoline"], label="بنزين فعلي", marker="o", color="red")
plt.plot(df["Year"], df["Diesel"], label="ديزل فعلي", marker="o", color="blue")
plt.plot(future_years, pred_gasoline, 'ro--', label="توقع بنزين")
plt.plot(future_years, pred_diesel, 'bo--', label="توقع ديزل")

plt.title("توقع أسعار الوقود في الجزائر")
plt.xlabel("السنة")
plt.ylabel("السعر (دج/لتر)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

























