using Distributions
using Parameters
using JuMP, GLPK
using LinearAlgebra
import Random
using Statistics
using Gadfly
import Cairo
# import Fontconfig

@with_kw mutable struct OnlineKnapsack
    T::Int64 = 1 # time horizon
    C::Float32 = 1f0 # Knapsack capacity
    item_sizes::Array{Float32,1} = Float32[] # item size
    item_values::Array{Float32,1} = Float32[] # item value
    lambda_sizes::Float64 = 1f0
    lambda_values::Float64 = 1f0
    # S # state space (assumes 1:nstates)
    A::Array{Int64, 1} = [1,2] # action space (assumes 1:nactions)
    S::Array{Int64,1} = 1:T+1 |> collect #state space
    γ = 0.95 # discount
    α = 0.2 # learning rate
    Q::Array{Float32,2} = zeros(length(S),2) # action value function
end

function build_env!(OK::OnlineKnapsack)
    for t = 1:OK.T+1
        push!(OK.item_sizes, rand(Exponential(OK.lambda_sizes)))
        push!(OK.item_values, rand(Exponential(OK.lambda_values)))
        # push!(OK.item_sizes, rand(Truncated(Normal(OK.lambda_sizes,10),0,Inf)))
        # push!(OK.item_values, rand(Truncated(Normal(OK.lambda_values,10),0,Inf)))
        # push!(OK.item_sizes, rand(Poisson(OK.lambda_sizes)))
        # push!(OK.item_values, rand(Poisson(OK.lambda_values)))
    end
end

#Linear Programming
function solve(OK::OnlineKnapsack)
    T, item_values, item_sizes, C = OK.T, OK.item_values, OK.item_sizes, OK.C
    model = Model(GLPK.Optimizer)
    @variable(model, x[1:T], Bin)
    @objective(model, Max, dot(x, item_values[1:T]))
    @constraint(model, dot(item_sizes[1:T], x) <= C)
    optimize!(model)
    return value.(x), objective_value(model)
end

function get_state(OK::OnlineKnapsack,
	t::Int64;
	remaining_capacity::Float32,
	selected_items::Array{Int64,1},
    rejected_items::Array{Int64,1}
)
	state = vcat(
		[OK.T],
		[remaining_capacity],
		selected_items,
        rejected_items,
		OK.item_sizes[1:t],
		OK.item_values[1:t],
        [t]
	)
	return state
end

mutable struct EpsilonGreedyExploration
    ϵ # probability of random arm
    α # exploration decay factor
end
function (π::EpsilonGreedyExploration)(OK::OnlineKnapsack, s, filtered_actions)
    t = convert(Int64, s[end])
    if rand() < π.ϵ
        π.ϵ *= π.α
        return rand(filtered_actions)
    else
        # return argmax(OK.Q[s,:])
        return maximum((OK.Q[t, i],i) for i in filtered_actions)[2]

    end
end

function update!(model::OnlineKnapsack, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    s = convert(Int64,s[end])
    s′= convert(Int64,s′[end])
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

function greedy(OK::OnlineKnapsack)
    sorted = sortperm(OK.item_values[1:OK.T]./OK.item_sizes[1:OK.T], rev=true)
    accum = 0
    x = zeros(OK.T)
    for i in sorted
        accum += OK.item_sizes[i]
        if accum <= OK.C
            x[i] = 1
        else
            accum -= OK.item_sizes[i]
        end
    end
    return dot(x, OK.item_values[1:OK.T])
end

function main(T_, α_, ϵ_, NUM_EPISODES_)
    NUM_AVG = 20
    list_optimal=zeros(NUM_AVG)
    list_qvalue=zeros(NUM_AVG)
    list_greedyvalue=zeros(NUM_AVG)
    list_current_reward = zeros(NUM_AVG, NUM_EPISODES_)
    println("$T_ Number of items")
    list_avg_100_scores = []
    OK = OnlineKnapsack(T = T_, α = α_, γ=0.95, C = 20f0, lambda_sizes = 10f0, lambda_values = 15f0)
    build_env!(OK)
    println("OK Env:", OK)

    for avg in 1:NUM_AVG
        if avg % (NUM_AVG/2) == 0
        	println("Average number $avg")
        end
        #Optimal
        OK = OnlineKnapsack(T = T_, α = α_, γ=0.95, C = 20f0, lambda_sizes = 10f0, lambda_values = 15f0)
        build_env!(OK)
        # println("OK Env:", OK)
        optimal_policy, optimal_value = solve(OK)
        # push!(list_optimal, optimal_value)
        list_optimal[avg]=optimal_value
        greedy_value = greedy(OK)
        list_greedyvalue[avg]=greedy_value
        #Q_value
        π = EpsilonGreedyExploration(ϵ_, 0.99)
        T, item_values, item_sizes, C = OK.T, OK.item_values, OK.item_sizes, OK.C
        global scores = []
        global avg_100_scores = []
        for ep in 1:NUM_EPISODES_
            capacity = deepcopy(C)
            selected_items=Int64[]
            s_items=zeros(Int64, T)
            r_items=zeros(Int64, T)
            reward = 0f0
            if ep % (NUM_EPISODES_/2) == 0
        		# println("Episode $ep Training")
            end
            for t in 1:OK.T
                state = get_state(OK,
                	t,
                	remaining_capacity=capacity,
                	selected_items=s_items,
                    rejected_items=r_items,
                )
                filtered_actions = deepcopy(OK.A)
                if capacity<item_sizes[t]
                    deleteat!(filtered_actions, 2)
                end
                action = π(OK, state, filtered_actions)

                if action==OK.A[1]
                    r_items[t]=1
                end

                if action==OK.A[2]
                    s_items[t]=1
                    capacity -= item_sizes[t]
                    reward += OK.item_values[t]
                end

                next_state = get_state(OK,
                	t+1,
                	remaining_capacity=capacity,
                	selected_items=s_items,
                    rejected_items=r_items,
                )
                update!(OK, state, action, reward, next_state)
                # function update!(model::OnlineKnapsack, s, a, r, s′)
                #     γ, Q, α = model.γ, model.Q, model.α
                #     Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
                #     return model
                # end
            end
            # list_current_reward[avg, ep]=reward
            push!(scores, reward)

            if ep > 100
                last_100_mean = mean(scores[end-99:end])
                push!(avg_100_scores, last_100_mean)
            end
            list_current_reward[avg, ep] = reward
        end
        # println("size",size(list_avg_100_scores))
        Q_reward=0
        Q_policy=[]
        for t in 1:OK.T
            action = argmax(OK.Q[t, :])
            # push!(Q_policy, action-1)
            if action==2
                Q_reward+=item_values[t]
            end
        end
        # Q_reward = dot(push!(Q_policy, 0), OK.item_values)
        list_qvalue[avg]=Q_reward
    end
    #mean(list_current_reward, dims=1)
    return mean(list_optimal, dims=1), mean(list_qvalue, dims=1), mean(list_greedyvalue, dims=1), mean(list_current_reward, dims=1)
end

#Plot Reward vs time horizon (=number of items)
Tlist = 10:10:50
plot_opt = zeros(length(Tlist))
plot_qv = zeros(length(Tlist))
plot_gv = zeros(length(Tlist))
for t in 1:length(Tlist)
    opt, qv, gv, _ = main(Tlist[t], 0.2, 0.3, 1)
    plot_opt[t] = opt[1]
    plot_qv[t] = qv[1]
    plot_gv[t] = gv[1]
end

p = plot(layer(x=Tlist, y=plot_opt, Geom.line, Geom.point, Theme(default_color=colorant"red")),
layer(x=Tlist, y=plot_qv, Geom.line, Geom.point),
layer(x=Tlist, y=plot_gv, Geom.line, Geom.point, Theme(default_color=colorant"yellow")),
Guide.XLabel("Number of items"), Guide.YLabel("Total Rewards"),
Guide.Title("Total Rewards vs Number of Items"),
Guide.manual_color_key("Legend", ["Optimal Reward", "Q Reward", "Greedy"], ["red", "deepskyblue", "yellow"]))
img = PNG("Reward-timehorizon-250.png")
draw(img, p)
display(p)

#Reward throughout Episode
NUM_EPISODES_ = 8_000
list_EPISODES = 1:NUM_EPISODES_
plot_cr = main(20, 0.2, 0.3, NUM_EPISODES_)[4]
p2 = plot(layer(x=list_EPISODES, y=plot_cr, Geom.line, Geom.point, Theme(default_color=colorant"red")),
Guide.XLabel("Episodes"), Guide.YLabel("Rewards"),
Guide.Title("Rewards vs Episodes"),
Guide.manual_color_key("Legend", ["Reward with ϵ=0.3"], ["red"]))
img2 = PNG("Reward-episode-eps=0.3.png")
draw(img2, p2)
display(p2)
#
# plot_cr = main(20, 0.2, 0.5, NUM_EPISODES_)[4]
# p2 = plot(layer(x=list_EPISODES, y=plot_cr, Geom.line, Geom.point, Theme(default_color=colorant"red")),
# Guide.XLabel("Episodes"), Guide.YLabel("Rewards"),
# Guide.Title("Rewards vs Episodes"),
# Guide.manual_color_key("Legend", ["Reward with ϵ=0.5"], ["red"]))
# img2 = PNG("Reward-episode-eps=0.5.png")
# draw(img2, p2)
# display(p2)
#
# plot_cr = main(20, 0.2, 0.9, NUM_EPISODES_)[4]
# p2 = plot(layer(x=list_EPISODES, y=plot_cr, Geom.line, Geom.point, Theme(default_color=colorant"red")),
# Guide.XLabel("Episodes"), Guide.YLabel("Rewards"),
# Guide.Title("Rewards vs Episodes"),
# Guide.manual_color_key("Legend", ["Reward with ϵ=0.9"], ["red"]))
# img2 = PNG("Reward-episode-eps=0.9.png")
# draw(img2, p2)
# display(p2)
