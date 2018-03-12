l_one = []
l_two = []

# create a list:
for x in range(1, 101, 3):
	l_one.append(x)

for x in range(0, 100, 3):
	l_two.append(x)

# print List
print(l_one)
print(l_two)

# iteret list
for so in l_one:
	print("one:", so),
	for st in l_two:
		print("two:", st)
