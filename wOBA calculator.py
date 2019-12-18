print("This is a 2018 wOBA calculator using weights from the 2018 MLB season")
print(" ")
player = input("Enter the player name: ")
BB = eval(input("Number of walks in 2018: "))
HBP = eval(input("Number of hit by pitches in 2018: "))
B1 = eval(input("Number of singles in 2018: "))
B2 = eval(input("Number of doubles in 2018: "))
B3 = eval(input("Number of triples in 2018: "))
HR = eval(input("Number of home runs in 2018: "))
AB = eval(input("Number of at-bats in 2018: "))
SF = eval(input("Number of sacrifice flies in 2018: "))
IBB = eval(input("Number of intentional walks in 2018: "))
uBB = BB - IBB
wOBA = (.690 * uBB + .720 * HBP + 0.880 * B1 + 1.247 * B2 + 1.578 * B3 +
2.031 * HR) / (AB + uBB + SF + HBP)
print("wOBA of", player, "is", round(wOBA, 3))




