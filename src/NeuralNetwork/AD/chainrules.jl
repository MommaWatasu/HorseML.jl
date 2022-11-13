using ChainRulesCore: RuleConfig, HasReverseMode, NoForwardsMode, AbstractThunk
struct HorseRuleConfig{CTX} <: RuleConfig{Union{HasReverseMode, NoForwardsMode}}
    context::CTX
end
HorseRuleConfig() = HorseRuleConfig(Context())

_is_rrule_redispatcher(m::Method) = m.sig == Tuple{typeof(rrule), RuleConfig, Vararg}

function has_chain_rrule(T)
    config_T , arg_Ts = Iterators.peel(T.parameters)
    configured_rrule_m = meta(Tuple{typeof(rrule), config_T, arg_Ts...})
    if _is_rrule_redispatcher(configured_rrule_m.method)
        rrule_m = meta(Tuple{typeof(rrule), arg_Ts...})
        no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), arg_Ts...})
    else
        rrule_m = configured_rrule_m
        no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), config_T, arg_Ts...})
    end
    
    do_not_use_rrule = matching_cr_sig(no_rrule_m, rrule_m)
    if do_not_use_rrule
        return false, configured_rrule_m.instance
    else
        return true, nothing
    end
end

matching_cr_sig(t, s) = matching_cr_sig(t.method.sig, s.method.sig)
matching_cr_sig(::DataType, ::UnionAll) = false
matching_cr_sig(::UnionAll, ::DataType) = false
matching_cr_sig(t::Type, s::Type) = type_tuple_tail(t) == type_tuple_tail(s)
matching_cr_sig(::Any, ::Nothing) = false

type_tuple_tail(d::DataType) = Tuple{d.parameters[2:end]...}
function type_tuple_tail(d::UnionAll)
    body = Base.unwrap_unionall(d)
    body_tt = type_tuple_tail(body)
    return Base.rewrap_unionall(body_tt, d)
end

"""
    wrap_chainrules_output(x)

Convert `x` from the differentials types ChainRules uses to the format Zygote uses internally.
"""
@inline wrap_chainrules_output(x) = x
@inline wrap_chainrules_output(x::AbstractThunk) = wrap_chainrules_output(unthunk(x))  # For now we are just not going to deal with thunks
@inline wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
# Zygote convention: even if many AbstractZero partials (i.e. multi-input function), make just 1 nothing.
@inline wrap_chainrules_output(x::Tuple{Vararg{ChainRules.AbstractZero}}) = nothing
@inline wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
@inline wrap_chainrules_output(x::ChainRulesCore.NotImplemented) = nothing
for T_outer in (:Tuple, :NamedTuple)
  # we create separate methods rather than using a `Union` + an `if` so that we avoid a
  # branch that changes output type, because nested AD on that kinda thing makes Zygote less
  # than happy.
  @eval @inline function wrap_chainrules_output(x::ChainRules.Tangent{P, T}) where {P, T<:$T_outer}
    xp = map(wrap_chainrules_output, canonicalize(x))
    ChainRulesCore.backing(xp)  # this is accessing ChainRulesCore internals, but it is prob safe enough, and it is fastest
  end
end
wrap_chainrules_output(dxs::AbstractArray{<:Number}) = dxs
wrap_chainrules_output(dxs::AbstractArray{<:AbstractArray{<:Number}}) = dxs
wrap_chainrules_output(dxs::AbstractArray) = map(wrap_chainrules_output, dxs)

"""
    wrap_chainrules_input(dx)

Convert `dx` from the format HorseML uses internally to differentials types ChainRules uses.
"""
@inline wrap_chainrules_input(dx) = dx
@inline wrap_chainrules_input(::Nothing) = ChainRules.ZeroTangent()
@inline wrap_chainrules_input(::Tuple{Vararg{Nothing}}) = ChainRules.ZeroTangent()
@inline wrap_chainrules_input(::AbstractArray{Nothing}) = ChainRules.ZeroTangent()
@inline function wrap_chainrules_input(dxs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, dxs)
  # This produces Tangent{Any} since it does not get to see the primal, `x`.
  ChainRulesCore.Tangent{Any, typeof(xp)}(xp)
end
# For mutable types, including x=Ref(1), Zygote makes Ref{Any}(::NamedTuple)
@inline wrap_chainrules_input(dx::Ref) = wrap_chainrules_input(dx[])
# For arrays, whitelist the safe ones, but always look inside Any[]:
@inline wrap_chainrules_input(dxs::AbstractArray{<:Number}) = dxs
@inline wrap_chainrules_input(dxs::AbstractArray{<:AbstractArray{<:Number}}) = dxs
@inline wrap_chainrules_input(dxs::AbstractArray) = map(wrap_chainrules_input, dxs)

@inline function _project(x, dx)
    wrap_chainrules_output(ProjectTo(horse2differential(dx, x)))
end

"""
  HBack{F}(back) <: Function

Wrapper for a ChainRules pullback `back`, that causes it to follow Horse conventions.
(A functor here is used rather than a closure to avoid boxing issues);
"""
struct HBack{F} <: Function
  back::F
end
@inline (s::HBack)(dy) = wrap_chainrules_output(s.back(wrap_chainrules_input(dy)))
# though it might be worth keeping as a performance optimization (benchmarking pending)
@inline (s::HBack)(::Nothing) = nothing

"""
    chain_rrule(config, f, args...)

Returns a the (primal) value of `f(args...)` and a pullback, by invoking `ChainRulesCore.rrule(f, args...)`.
The pullback is appropriately wrapped up to follow Horse conventions.
"""
@inline function chain_rrule(config, f, args...)
  y, back = rrule(config, f, args...)
  return y, HBack(back)
end


"""
  chain_rrule_kw(config, kwf, kwargs, f, args...)

As per [`chain_rrule`](@ref) but with support for kwargs.
`kwf` should be the kwfunc matching to `f`, and `kwargs` are a `NamedTuple` of keyword arguments.
"""
@inline function chain_rrule_kw(config, kwf, kwargs, f, args...)
  y, back = rrule(config, f, args...; kwargs...)
  function kw_hpullback(dy)
    dxs = HBack(back)(dy)
    if dxs === nothing  # if dxs is nothing, then all partiaols are nothing
      # Horse convention is a single nothing no mather how partials, if all are nothing
      return nothing
    else
      return (nothing, nothing, dxs...)  # first two nothings are for kwfunc and kwargs
    end
  end
  return y, kw_hpullback
end