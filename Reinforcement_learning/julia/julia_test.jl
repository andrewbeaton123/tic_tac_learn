matrix = [1 2 3; 4 5 6; 7 8 9]

# Number of rows and columns in the matrix
n = size(matrix, 1)

# Main diagonal elements
main_diagonal = matrix[1:n+1:end]

# Opposite diagonal elements
opposite_diagonal = matrix[n:n-1:end-n+1]

println("Main diagonal: ", main_diagonal)
println("Opposite diagonal: ", opposite_diagonal)
