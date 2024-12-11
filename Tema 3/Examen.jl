# using Pkg
# Pkg.activate(".")
# using Plots # Para representación de funciones
# include("Search.jl")

# function mover(s, mv)
#     x,y,c = s
#     i, j, k = mv
#     if abs(i-j)==1
#         k = c +1 
#     elseif abs(i-j) == 10 
#         k = c + 2 
#     else
#         k = c + 3
#     return (j, 0, k)
# end

# function valido(s, p)
    
#     x,y,c= s 
#     return (y != p and (abs(x-y) == 1 or abs(x-y )== 10 or abs(x-y) == 100))
# end

# function sucesoresMisioneros(s)
#     res = []
#     # Posibles traslados (max 2 personas en la barca) en cualquier dirección
#     moves = [(m, c) for m in 0:2 for c in 0:2 if 1 <= m + c <= 2]
#     for mv in moves
#         s1 = mover(s, mv)  # Aplicación del movimiento mv
#         # Comprobación de que el estado es válido
#         if valido(s1)
#             push!(res, s1)
#         end
#     end
#     return res
# end