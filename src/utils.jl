# PooledString is an internal-only type for efficiently tracking string data + length
# all strings indexed from a column/row will always be a full String
# specifically, it allows avoiding materializing full Strings for pooled string columns while parsing
# and allows a fastpath for materializing a full String when no escaping is needed
export PooledString
struct PooledString <: AbstractString end

struct PointerString <: AbstractString
    ptr::Ptr{UInt8}
    len::Int
end

function Base.hash(s::PointerString, h::UInt)
    h += Base.memhash_seed
    ccall(Base.memhash, UInt, (Ptr{UInt8}, Csize_t, UInt32), s.ptr, s.len, h % UInt32) + h
end

import Base: ==
function ==(x::String, y::PointerString)
    sizeof(x) == y.len && ccall(:memcmp, Cint, (Ptr{UInt8}, Ptr{UInt8}, Csize_t), pointer(x), y.ptr, y.len) == 0
end
==(y::PointerString, x::String) = x == y

Base.ncodeunits(s::PointerString) = s.len
@inline function Base.codeunit(s::PointerString, i::Integer)
    @boundscheck checkbounds(s, i)
    GC.@preserve s unsafe_load(s.ptr + i - 1)
end
Base.String(x::PointerString) = unsafe_string(x.ptr, x.len)

# default value to signal that parsing should try to detect a type
struct Empty end

# MISSING is a mask that can be combined w/ any other TypeCode for Union{T, Missing}
missingtype(T) = T >: Missing
usermissing(T) = T === Union{User, Missing}

# MISSINGTYPE is for a column like `Vector{Missing}`
# if we're trying to detect a column type and the 1st value of a column is `missing`
# we basically want to still treat it like EMPTY and try parsing other types on each row
struct MissingType end

# enum-like type codes for basic supported types
stringtype(T) = T >: String
pooled(T) = T >: PooledString

# a user-provided type; a mask that can be combined w/ basic types
struct User end
user(T) = T >: User

typebits(T) = Core.Compiler.typesubtract(gettype(T), Missing)
gettype(T) = Core.Compiler.typesubtract(T, User)

mutable struct SingleValueVector{T} <: AbstractVector{T}
    len::Int64
end

Base.IndexStyle(::Type{SingleValueVector}) = Base.IndexLinear()
Base.size(x::SingleValueVector) = (x.len,)
singlevalue(T) = missing
Base.getindex(x::SingleValueVector{T}, i::Int) where {T} = singlevalue(T)
Base.setindex!(x::SingleValueVector, val, i::Int) = nothing
Base.resize!(x::SingleValueVector, len) = x.len = len

allocate(::Type{T}, len::Int) where {T <: Union{Union{User, Missing}, Empty, MissingType}}= SingleValueVector{T}(len)
allocate(T, len::Int) = SentinelVector{typebits(T)}(undef, len)

using SentinelArrays
const ArrayType{T} = SentinelArray{T, 1, T, Missing, Vector{T}}

@inline function promote_typecode(T, S)
    if T === Empty || T === S || user(T) || nonmissingtype(T) === nonmissingtype(S)
        return Union{T, S}
    elseif T === MissingType
        return Union{S, Missing}
    elseif S === MissingType
        return Union{T, Missing}
    elseif T <: Integer
        return S <: AbstractFloat ? S : S <: Union{AbstractFloat, Missing} ? S : missingtype(S) ? Union{String, Missing} : String
    elseif T <: Union{Integer, Missing}
        return S <: AbstractFloat || S <: Union{AbstractFloat, Missing} ? Union{S, Missing} : Union{String, Missing}
    elseif T <: AbstractFloat
        return S <: Integer ? T : S <: Union{Integer, Missing} ? Union{T, Missing} : missingtype(S) ? Union{String, Missing} : String
    elseif T <: Union{AbstractFloat, Missing}
        return S <: Integer || S <: Union{Integer, Missing} ? Union{T, Missing} : Union{String, Missing}
    elseif missingtype(T) || missingtype(S)
        return Union{String, Missing}
    else
        return String
    end
end

# bit patterns for missing value, int value, escaped string, position and len in tape parsing
const PosLen = UInt64

# primitive type PosLen 64 end
# PosLen(x::UInt64) = Core.bitcast(PosLen, x)
# UInt64(x::PosLen) = Core.bitcast(UInt64, x)

const MISSING_BIT = 0x8000000000000000
missingvalue(x) = (UInt64(x) & MISSING_BIT) == MISSING_BIT

const ESCAPE_BIT = 0x4000000000000000
escapedvalue(x) = (UInt64(x) & ESCAPE_BIT) == ESCAPE_BIT

getpos(x) = (UInt64(x) & 0x3ffffffffff00000) >> 20
getlen(x) = UInt64(x) & 0x00000000000fffff

_unsafe_string(p, len) = ccall(:jl_pchar_to_string, Ref{String}, (Ptr{UInt8}, Int), p, len)
# TODO: handle escaping
str(::Type{String}, buf, poslen) = _unsafe_string(pointer(buf, getpos(poslen)), getlen(poslen))
str(::Type{String}, buf, pos, len) = _unsafe_string(pointer(buf, pos), len)

str(::Type{PosLen}, buf, poslen) = poslen

const Str = PosLen

# one-liner suggested from ScottPJones
consumeBOM(buf) = (length(buf) >= 3 && buf[1] == 0xef && buf[2] == 0xbb && buf[3] == 0xbf) ? 4 : 1

function slurp(source)
    N = 2^22 # 4 mb increments
    ms = Tuple{Vector{UInt8}, Int}[]
    tot = 0
    while true
        m = Mmap.mmap(Vector{UInt8}, N)
        n = readbytes!(source, m, N)
        tot += n
        push!(ms, (m, n))
        n < N && break
    end
    final = Mmap.mmap(Vector{UInt8}, tot)
    pos = 1
    for (m, n) in ms
        copyto!(final, pos, m, 1, n)
        pos += n
    end
    return final
end

getsource(source::Vector{UInt8}, ::Any) = source
getsource(source::Cmd, ::Any) = Base.read(source)
getsource(source::AbstractPath, ::Any) = Base.read(open(source))
getsource(source::IO, ::Any) = slurp(source)
getsource(source::SystemPath, use_mmap) = getsource(string(source), use_mmap)
function getsource(source, use_mmap)
    m = Mmap.mmap(source)
    if use_mmap
        return m
    end
    m2 = Mmap.mmap(Vector{UInt8}, length(m))
    copyto!(m2, 1, m, 1, length(m))
    finalize(m)
    return m2
end

getname(buf::Vector{UInt8}) = "<raw buffer>"
getname(cmd::Cmd) = string(cmd)
getname(str) = string(str)
getname(io::I) where {I <: IO} = string("<", I, ">")

# normalizing column name utilities
const RESERVED = Set(["local", "global", "export", "let",
    "for", "struct", "while", "const", "continue", "import",
    "function", "if", "else", "try", "begin", "break", "catch",
    "return", "using", "baremodule", "macro", "finally",
    "module", "elseif", "end", "quote", "do"])

normalizename(name::Symbol) = name
function normalizename(name::String)::Symbol
    uname = strip(Unicode.normalize(name))
    id = Base.isidentifier(uname) ? uname : map(c->Base.is_id_char(c) ? c : '_', uname)
    cleansed = string((isempty(id) || !Base.is_id_start_char(id[1]) || id in RESERVED) ? "_" : "", id)
    return Symbol(replace(cleansed, r"(_)\1+"=>"_"))
end

function makeunique(names)
    set = Set(names)
    length(set) == length(names) && return Symbol[Symbol(x) for x in names]
    nms = Symbol[]
    for nm in names
        if nm in nms
            k = 1
            newnm = Symbol("$(nm)_$k")
            while newnm in set || newnm in nms
                k += 1
                newnm = Symbol("$(nm)_$k")
            end
            nm = newnm
        end
        push!(nms, nm)
    end
    return nms
end

standardize(::Type{T}) where {T <: Integer} = Int64
standardize(::Type{T}) where {T <: Real} = Float64
standardize(::Type{T}) where {T <: Dates.TimeType} = T
standardize(::Type{Bool}) = Bool
standardize(::Type{PooledString}) = PooledString
standardize(::Type{<:AbstractString}) = String
standardize(T) = Empty

initialtypes(T, x::AbstractDict{String}, names) = Type[haskey(x, string(nm)) ? Union{standardize(x[string(nm)]), User} : T for nm in names]
initialtypes(T, x::AbstractDict{Symbol}, names) = Type[haskey(x, nm) ? Union{standardize(x[nm]), User} : T for nm in names]
initialtypes(T, x::AbstractDict{Int}, names)    = Type[haskey(x, i) ? Union{standardize(x[i]), User} : T for i = 1:length(names)]

# given a DateFormat, is it meant for parsing Date, DateTime, or Time?
function timetype(df::Dates.DateFormat)
    date = false
    time = false
    for token in df.tokens
        T = typeof(token)
        if T in (Dates.DatePart{'H'}, Dates.DatePart{'I'}, Dates.DatePart{'M'}, Dates.DatePart{'S'}, Dates.DatePart{'s'})
            time = true
        elseif T in (Dates.DatePart{'y'}, Dates.DatePart{'Y'}, Dates.DatePart{'m'}, Dates.DatePart{'d'}, Dates.DatePart{'u'}, Dates.DatePart{'U'})
            date = true
        end
    end
    return ifelse(date & time, DateTime, ifelse(time, Time, Date))
end

# if a cell value of a csv file has escape characters, we need to unescape it
function unescape(s, e)
    n = ncodeunits(s)
    buf = Base.StringVector(n)
    len = 1
    i = 1
    @inbounds begin
        while i <= n
            b = codeunit(s, i)
            if b == e
                i += 1
                b = codeunit(s, i)
            end
            @inbounds buf[len] = b
            len += 1
            i += 1
        end
    end
    resize!(buf, len - 1)
    return String(buf)
end

"""
    CSV.detect(str::String)

Use the same logic used by `CSV.File` to detect column types, to parse a value from a plain string.
This can be useful in conjunction with the `CSV.Rows` type, which returns each cell of a file as a String.
The order of types attempted is: `Int64`, `Float64`, `Date`, `DateTime`, `Bool`, and if all fail, the input String is returned.
No errors are thrown.
For advanced usage, you can pass your own `Parsers.Options` type as a keyword argument `option=ops` for sentinel value detection.
"""
function detect end

detect(str::String; options=Parsers.OPTIONS) = something(detect(codeunits(str), 1, sizeof(str), options), str)

function detect(buf, pos, len, options)
    int, code, vpos, vlen, tlen = Parsers.xparse(Int64, buf, pos, len, options)
    if Parsers.sentinel(code) && code > 0
        return missing
    end
    if Parsers.ok(code) && vpos + vlen - 1 == len
        return int
    end
    float, code, vpos, vlen, tlen = Parsers.xparse(Float64, buf, pos, len, options)
    if Parsers.ok(code) && vpos + vlen - 1 == len
        return float
    end
    if options.dateformat === nothing
        try
            date, code, vpos, vlen, tlen = Parsers.xparse(Date, buf, pos, len, options)
            if Parsers.ok(code) && vpos + vlen - 1 == len
                return date
            end
        catch e
        end
        try
            datetime, code, vpos, vlen, tlen = Parsers.xparse(DateTime, buf, pos, len, options)
            if Parsers.ok(code) && vpos + vlen - 1 == len
                return datetime
            end
        catch e
        end
    else
        try
            # use user-provided dateformat
            DT = timetype(options.dateformat)
            dt, code, vpos, vlen, tlen = Parsers.xparse(DT, buf, pos, len, options)
            if Parsers.ok(code) && vpos + vlen - 1 == len
                return dt
            end
        catch e
        end
    end
    bool, code, vpos, vlen, tlen = Parsers.xparse(Bool, buf, pos, len, options)
    if Parsers.ok(code) && vpos + vlen - 1 == len
        return bool
    end
    return nothing
end

struct ReversedBuf <: AbstractVector{UInt8}
    buf::Vector{UInt8}
end

Base.size(a::ReversedBuf) = size(a.buf)
Base.IndexStyle(::Type{ReversedBuf}) = Base.IndexLinear()
Base.getindex(a::ReversedBuf, i::Int) = a.buf[end + 1 - i]

function unset!(A::Vector, i::Int, row, x)
    finalize(A[i])
    ccall(:jl_arrayunset, Cvoid, (Array, Csize_t), A, i - 1)
    # println("deleting col = $i on thread = $(Threads.threadid()), row = $row, id = $x")
    return
end

memcpy!(d, doff, s, soff, n) = ccall(:memcpy, Cvoid, (Ptr{UInt8}, Ptr{UInt8}, Int), d + doff - 1, s + soff - 1, n)