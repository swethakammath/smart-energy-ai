
# Smart Energy AI
# Step 5: Appliance ON/OFF Detection and Energy Calculation

print("================================")
print("       SMART ENERGY AI")
print("================================")

# Appliance power ratings (Watts)
heater = 2000
oven = 1800
washing_machine = 500

# Display appliance power
print("\nAppliance Power:")
print("Heater:", heater, "W")
print("Oven:", oven, "W")
print("Washing Machine:", washing_machine, "W")

# Check which appliances are ON
print("\nCheck Appliance Status")

heater_on = input("Is heater ON? (y/n): ").lower()
oven_on = input("Is oven ON? (y/n): ").lower()
washing_machine_on = input(
    "Is washing machine ON? (y/n): "
).lower()

# Initialize power and energy
total_power = 0
total_energy = 0

# Electricity rate
rate = float(input("\nEnter electricity rate (₹ per unit): "))

# Heater calculation
heater_energy = 0

if heater_on == "y":
    total_power += heater

    heater_hours = float(
        input("Enter heater usage hours: ")
    )

    heater_energy = (heater * heater_hours) / 1000
    total_energy += heater_energy

# Oven calculation
oven_energy = 0

if oven_on == "y":
    total_power += oven

    oven_hours = float(
        input("Enter oven usage hours: ")
    )

    oven_energy = (oven * oven_hours) / 1000
    total_energy += oven_energy

# Washing machine calculation
washing_machine_energy = 0

if washing_machine_on == "y":
    total_power += washing_machine

    washing_machine_hours = float(
        input("Enter washing machine usage hours: ")
    )

    washing_machine_energy = (
        washing_machine * washing_machine_hours
    ) / 1000

    total_energy += washing_machine_energy

# Calculate electricity cost
total_cost = total_energy * rate

# Power limit
safe_limit = 3500

# Display results
print("\n================================")
print("          RESULTS")
print("================================")

print("Heater Energy:", round(heater_energy, 2), "kWh")
print("Oven Energy:", round(oven_energy, 2), "kWh")
print(
    "Washing Machine Energy:",
    round(washing_machine_energy, 2),
    "kWh"
)

print("\nTotal Power:", total_power, "W")
print("Total Energy:", round(total_energy, 2), "kWh")
print("Estimated Electricity Cost: ₹", round(total_cost, 2))

# High-power warning
if total_power > safe_limit:
    print("\n⚠️ HIGH POWER USAGE!")
    print("Warning: Power exceeds the configured limit.")
else:
    print("\nPower usage is within the configured limit.")

print("\nThank you for using Smart Energy AI!")