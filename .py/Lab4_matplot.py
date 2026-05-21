import matplotlib.pyplot as plt
# x = [1,2,9,4,7]
# y = [4,5,8,7,12]
# plt.plot(x,y,c='red',ls= '-.',lw= 5)
# plt.title('Line Plot')
# plt.xlabel('x-axis')
# plt.ylabel('this is y axis')
# plt.grid()
# plt.show()
#########################

# x = [6,4,12,7,7]
# y = [4,5,8,7,10]
# plt.scatter(x,y,c='green')
# plt.title('Scatter Plot')
# plt.xlabel('x-axis')
# plt.grid()
# plt.ylabel('y axis')
# plt.show()
##############################
# x = [5,2,9,4,7]
# y = [10,5,8,4,2]
# plt.bar(x,y)
# plt.title('Bar Chart')
# plt.xlabel('values')
# plt.grid()
# plt.ylabel('frequencies')
# plt.show()
###############################
# y = [10,5,8,4,10,5,5]
# plt.hist(y)
# plt.title('Histogram')
# plt.xlabel('values')
# plt.grid()
# plt.ylabel('frequencies')
# plt.show()
###############################

# x = [2, 2, 3.5, 3.5]
# y = [1, 2.5, 2.5, 1]
# plt.fill(x, y, color='blue')
#
# plt.xlim(0, 6)
# plt.ylim(0, 7)
# plt.axis('off')
# plt.show()

#################################
# Base of the house
x_base = [2, 2, 8, 8, 2]
y_base = [2, 6, 6, 2, 2]

# Roof of the house (triangle)
x_roof = [1, 5, 9, 1]
y_roof = [6, 9, 6, 6]

# Door of the house
x_door = [4, 4, 6, 6, 4]
y_door = [2, 4, 4, 2, 2]

# Plot the shapes
plt.plot(x_base, y_base, c="black")
plt.plot(x_roof, y_roof, c="black")
plt.plot(x_door, y_door, c="black")

# Set axis limits
plt.xlim(0, 10)
plt.ylim(0, 10)

# Hide axes
plt.axis('off')

# Show the plot
plt.show()
