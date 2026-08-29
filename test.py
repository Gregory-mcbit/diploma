print("X Y Z W F")

for x in range(2):
    for y in range(2):
        for z in range(2):
            for w in range(2):
                func = (not (y <= x)) or (z <= w) or (not z)

                if func == 0:
                    print(x, y, z, w, 0)
