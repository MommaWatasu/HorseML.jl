###This is internal processing###

#check whether `f` is keyword-function or not(`kw` = keyword)
is_kwfunc(::Vararg{Any}) = false
is_kwfunc(k, ::Type{<:NamedTuple}, f, args...) = k===Core.kwftype(f)

struct CompileError
    T
    e
end

function Base.showerror(io::IO, e::CompileError)
    print(io, "Compiling $(e.T): ")
    showerror(io, e.e)
end

mutable struct Context{I}
    cache::Union{IdDict{Any, Any}, Nothing}
end

Context() = Context{false}(nothing)
cache(ctx::Context) = ctx.cache === nothing ? (ctx.cache = IdDict()) : ctx.cache

struct Pullback{S, T}
    t::T
end

Pullback{S}(x) where S = Pullback{S, typeof(x)}(x)

ignore_sig(T) = all(T -> T <: Type, T.parameters)

function edge!(m::IRTools.Meta, edge::Core.MethodInstance)
  m.code.edges === nothing && (m.code.edges = Core.MethodInstance[])
  push!(m.code.edges, edge)
  return
end

@generated function _pullback(ctx::Context, f, args...)
    #Try using ChainRulesCore
    if is_kwfunc(f, args...)
        # if it kw then `args[1]` are the keyword args, `args[2]` is actual function
        cr_T = Tuple{HorseRuleConfig{ctx}, args[2:end]...}
        chain_rrule_f = :chain_rrule_kw
    else
        cr_T = Tuple{HorseRuleConfig{ctx}, f, args...}
        chain_rrule_f = :chain_rrule
    end
    
    hascr, cr_edge = has_chain_rrule(cr_T)
    hascr && return :($chain_rrule_f(HorseRuleConfig(ctx), f, args...))
    
    T = Tuple{f, args...}
    ignore_sig(T) && return :(f(args...), Pullback{$T}(()))
    
    g = try
        _generate_pullback_via_decomposition(T)
    catch e
        rethrow(CompileError(T, e))
    end
    g === nothing && return :(f(args...), Pullback{$T}(()))
    meta, forw, _ = g
    argnames!(meta, Symbol("#self#", :ctx, :f, :args))
    forw = varargs!(meta, forw, 3)
    forw = slots!(pis!(inlineable!(forw)))
    cr_edge != nothing && edge!(meta, cr_edge)
    return update!(meta.code, forw)
end

###End ot internal processing###

tailemaybe(::Nothing) = nothing
tailemaybe(x::Tuple) = Base.tail(x)

@inline pullback(f, args...) = pullback(Context(), f, args...)
function pullback(ctx::Context, f, args...)
    y, back = _pullback(ctx, f, args...)
    y, Δ -> tailemaybe(back(Δ))
end

#I have to add the process to make function certain
function gradient(f, args...)
    y, back = pullback(f, args...)
    grad = back(y)
    isnothing(grad) ? nothing : map(_project, args, grad)
end

Base.adjoint(f::Function) = x -> begin
    y, back = pullback(f, x)
    back(y)[1]
end