import json

with open("experiments_log.json", 'r') as f:
    logs = json.load(f)

# Ordenar por accuracy descendente
best = sorted(logs, key=lambda x: x['accuracy'], reverse=True)[0]

print(f"🏆 Mejor Modelo: {best['name']}")
print(f"📈 Accuracy: {best['accuracy']}")
print(f"⚙️  Params: {best['params']}")