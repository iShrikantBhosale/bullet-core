import bullet_ai
import sys

print("Module loaded:", bullet_ai)
print("Dir:", dir(bullet_ai))
print("BulletModel doc:", bullet_ai.BulletModel.__doc__)
try:
    print("BulletModel init doc:", bullet_ai.BulletModel.__init__.__doc__)
except:
    print("No __init__ doc")

print("Creating model with valid path...")
try:
    model = bullet_ai.BulletModel("tiny.bullet")
    print("Model created")
except Exception as e:
    print("Error:", e)
