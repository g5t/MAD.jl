module MAD
using PyPlot,easyClipper # for drawpatch, pathdifference, etc.
using Printf # for @sprintf
using LinearAlgebra, Statistics # for norm and mean
export q2a,a2q,q2xy
export k2E, E2k, kikf2E
export arcxy,MA7blockedkin,MA7blockedkfn
export drawpatch, drawMA7Qxy, drawMA7a3a4, drawQEpoints, drawTASPMA7

const ħ²2mₙ=2.0722 # meV Å²

"""
	q2a(a,alpha,va,vb,QE; fx=2,kfix=1.4,sense[-1,1,-1])

Calculates the angles A1-A6 according to MAD given a lattice and two vectors in the scattering plane.

|Inputs | Type | Description |
|:-----:|:----:|:------------|
| a    | `[as bs cs]` | a three-vector of lattice parameters                            |
| α    | `[aa bb cc]` | a three-vector of lattice angles                                |
| va   | `[ax ay az]` | one crystal vector in the plane in r.l.u.                       |
| vb   | `[ax ay az]` | the other crystal vector in the plane in r.l.u.                 |
| QE   | `[h k l {E}]` | a 3-vector of momentem transfer in r.l.u. and optional 4th value of energy transfer meV |
| fx   | `Number`     | `fx=1` for fixed ki, `fx=2` for fixed kf                        |
| kfix | `Number`     | the fixed wavevector in inverse Angstrom                        |
| sens | `[sm ss sa]` | the scattering senses at the monochromator, sample and analyzer |

|Outputs | Type | Description |
|:-----:|:----:|:------------|
| ang   | `[a1 a2 a3 a4 a5 a6]` | the TAS angles in degrees |

Original MATLAB version (mad_q2a.m) Duc Le

Translated to julia by Gregory Tucker
"""
function q2a(a::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Vector;fx=2,kfix=1.4,sens=[-1,1,-1])
  @assert length(a)==length(alpha)==length(va)==length(vb)==3
  @assert 5>length(QE)>2
  QE=copy(QE)
  length(QE)==3 && push!(QE,0)
  # Some constants
  Q_e=1.60217653e-19; hbar=1.05457168e-34; m_n=1.67492728e-27;
  # Calculates the transformation matrix
  a=a/2pi # if a./=2pi instead, the changes to a leak out of this function!
  aspv=hcat(va,vb);
  @assert !any(abs.(a).<eps()) "Lattice parameters cannot be zero"
  cosa=cosd.(alpha); sina=sind.(alpha)
  cc=1+2*prod(cosa)-sum(abs2,cosa) # Calculates the reciprocal lattice angles beta
  @assert cc>eps() "Passed lattice angles are wrong"
  cc=sqrt(cc)
  b = sina./(a*cc);
  cosb = [( cosa[2]*cosa[3] - cosa[1] ) / ( sina[2]*sina[3] );          # Based on t_rlp.for from MAD
          ( cosa[3]*cosa[1] - cosa[2] ) / ( sina[3]*sina[1] );
          ( cosa[1]*cosa[2] - cosa[3] ) / ( sina[1]*sina[2] )];
  sinb = sqrt.(1 .- cosb.^2);
  bb = [b[1] b[2]*cosb[3]  b[3]*cosb[2];                                #Calculates Busing and Levy's B matrix
        0    b[2]*sinb[3] -b[3]*sinb[2]*cosa[1];
        0    0             1/a[3]];
  rlb =hcat( transpose(sum(abs2,bb,dims=1)), abs.(atan.(sinb,cosb))*180/pi );
  @assert !any(abs.(rlb[:,1]).<eps()) "Error computing B matrix"

  vv = copy(transpose(bb*aspv)) # (3x3 * 3x2 ==> 3x2) => 2x3
  vv = vcat(vv,transpose(cross(vec(vv[1,:]),vec(vv[2,:]))))
  vv[2,:]= -transpose(cross(vec(vv[1,:]),vec(vv[3,:])))
  c = sum(abs2,vv,dims=2)
  @assert !any(abs.(c).<eps()) "Failed to calculate C"
  c = sqrt.(c);
  vv./=repeat(c,outer=[1,3])
  rl2sp = vv*bb; # 3x3 * 3x3 ==> 3x3

  # Calculates a1/a2, a5/a6
  if fx==1   # ki fixed
    ki=kfix; kf = sqrt( (( hbar^2*(kfix*1e10)^2/(2*m_n))/(Q_e/1000)-QE[4])*(2*m_n*Q_e/1000) )/hbar/1e10;
  else       # kf fixed
    kf=kfix; ki = sqrt( (( hbar^2*(kfix*1e10)^2/(2*m_n))/(Q_e/1000)+QE[4])*(2*m_n*Q_e/1000) )/hbar/1e10;
  end
  a1 = sens[1]*asin((2*pi/ki)/2/3.355)*180/pi; a2 = 2*a1;
  a5 = sens[3]*asin((2*pi/kf)/2/3.355)*180/pi; a6 = 2*a5;

  # Calculates a3/a4
  qt = rl2sp*QE[1:3];          # Vector in the scattering plane *should be* [qx,qy,0]
  mqt=norm(qt)
  arg = (ki^2+kf^2-mqt^2)/(2*ki*kf);  # two-theta
  abs(arg)≤1 || error("cos(a4) == $arg is unphysical, with a/2pi=$a, alpha=$alpha, va=$va, vb=$vb, QE=$QE")
  a4 = acos(arg)*sens[2]*180/pi;
  a3 = (-atan(qt[2],qt[1])-acos((kf^2-mqt^2-ki^2)/(-2*mqt*ki))*sign(a4))*180/pi;

  ang = [a1;a2;a3;a4;a5;a6];
end

"""
(Qx,Qy)=a2q(ki,a3,a4,kf)
A utility to calculate the x and y components of `Q` in inverse Angstrom given
the values of `ki`, `kf` in inverse Angstrom and `a3` and `a4` in degrees.
"""
function a2q(ki,a3::AbstractVector,a4::AbstractVector,kf)
	@assert length(a3)==length(a4)
	Q=sqrt.(ki^2+kf^2 .- 2*ki*kf*cosd.(a4))
	cosΨ = (kf^2-ki^2 .- Q.^2)./(-2*ki*Q)
	cosΨ[isnan.(cosΨ)].=zero(eltype(Q)) # protect against cosΨ = NaN when Q=0 (good for plotting, bad for driving an instrument)
	cosΨ[abs.(cosΨ).>1].=sign.(cosΨ[abs.(cosΨ).>1]) # protect against acos(>1)
	Ψ=sign.(a4).*acosd.(cosΨ)
	Qx= Q.*cosd.(a3.+Ψ)
	Qy=-Q.*sind.(a3.+Ψ) # match the definition used my TASMAD
	return (Qx,Qy)
end
function a2q(ki,a3::Number,a4::AbstractVector,kf)
	Q=sqrt.(ki^2+kf^2 .- 2*ki*kf*cosd.(a4))
	cosΨ = (kf^2-ki^2 .- Q.^2)./(-2*ki*Q)
	cosΨ[isnan.(cosΨ)].=zero(eltype(Q)) # protect against cosΨ = NaN when Q=0 (good for plotting, bad for driving an instrument)
	cosΨ[abs.(cosΨ).>1].=sign.(cosΨ[abs.(cosΨ).>1]) # protect against acos(>1)
	Ψ=sign.(a4).*acosd.(cosΨ)
	Qx= Q.*cosd.(a3.+Ψ)
	Qy=-Q.*sind.(a3.+Ψ) # match the definition used my TASMAD
	return (Qx,Qy)
end
function a2q(ki,a3::AbstractVector,a4::Number,kf)
	Q=sqrt(ki^2+kf^2-2*ki*kf*cosd(a4))
	cosΨ = (kf^2-ki^2-Q^2)/(-2*ki*Q)
	isnan(cosΨ) && (cosΨ=zero(typeof(Q)))
	abs(cosΨ)>1 && (cosΨ=sign(cosΨ))
	Ψ=sign(a4)*acosd(cosΨ)
	Qx= Q.*cosd.(a3.+Ψ)
	Qy=-Q.*sind.(a3.+Ψ) # match the definition used my TASMAD
	return (Qx,Qy)
end
function a2q(ki,a3::Number,a4::Number,kf)
	Q=sqrt(ki^2+kf^2-2*ki*kf*cosd(a4))
	cosΨ = (kf^2-ki^2-Q^2)/(-2*ki*Q)
	isnan(cosΨ) && (cosΨ=zero(typeof(Q)))
	abs(cosΨ)>1 && (cosΨ=sign(cosΨ))
	Ψ=sign(a4)*acosd(cosΨ)
	Qx= Q*cosd(a3+Ψ)
	Qy=-Q*sind(a3+Ψ) # match the definition used my TASMAD
	return (Qx,Qy)
end
a2q(ki,as::Tuple,kf)=a2q(ki,as[1],as[2],kf)
a2q(ki,a3,a4,kf)=tuple(collect.(collect(zip(a2q.(ki,a3,a4,kf)...)))...)


"""
	(x,y)=arcxy(r,θ; x0=0,y0=0,θ0=0,N=100)
A simple utility function to define the points of an arc with radius `r` covering
an arc length `θ` (in degrees) centered on a point (`x0`,`y0`) and covering the portion of a
circle from `θ0-θ/2` to `θ0+θ/2` with a total of `N` points along the path.
The output is in the form of a `Tuple` of the two vectors.
"""
arcxy(r,θ;x0=0,y0=0,θ0=0,N=100)=( t=θ0.+StepRangeLen(-θ/2,θ/(N-1),N); return (x0.+r*cosd.(t),y0.+r*sind.(t)) )


function outlinepath(points::Vector{T},N::R) where {T,R}
	P=Base.promote_op(/,T,R)
	l=length(points)
	outline=Array{P}(undef, l*(N-1) + 1)
	offset=0
	for i=1:l-1
		# outline[offset.+(1:N)]=range(points[i],stop=points[i+1],length=N) # THIS ISN'T TYPE STABLE FOR ARRAY{T} ELEMENTS!!!
		outline[offset.+(1:N)]=StepRangeLen(points[i], (points[i+1].-points[i])/(N-1), N)
		offset+=N-1
	end
	outline[offset.+(1:N)]=StepRangeLen(points[end], (points[1].-points[end])/(N-1), N)
	return outline::Vector{P}
end
"""
	points2vectors(::Vector{Vector|NTuple})
For an input vector of N-D points expressed as `Vector`s or `NTuple`s
`points2vectors` pulls together N vectors of the 1:N coordinates suitable for, e.g., plotting
"""
function points2vectors(p::Vector{T}) where T<:Vector
       v=hcat(p...)
       ntuple(i->v[i,:], size(v,1))
end
function points2vectors(p::Vector{NTuple{N,T}}) where {N,T}
    v=hcat([ [x...] for x in p]...)
    ntuple(i->v[i,:],N)
end
tuples2points(t::Vector{NTuple{N,T}}) where {N,T} = [[x...] for x in t]

"""drawpatch accesses the low-level matplotlib routines to add a patch to a figure"""
function drawpatch(x,y,close=true;kwds...)
  patch=PyPlot.matplotlib[:patches][:Polygon](hcat(x,y),close;kwds...)
  gca()[:add_patch](patch)
end
drawpatch(xy::Tuple,o...;kwds...)=drawpatch(xy[1],xy[2],o...;kwds...)

labfmt(x)=@sprintf("%.4f",x)


k2E(k)=ħ²2mₙ * k.^2
E2k(E)=sqrt.(E./ħ²2mₙ)
kikf2E(ki,kf)=k2E(ki)-k2E(kf)
k2a1(k)=asind.(π/3.355./k) # assuming we're worried about a PG(002) Bragg scattering device
k2a2(k)=2k2a1(k)

function q2xy(lengths::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Vector,ki,kf,sens=[-1,1,-1])
	@assert 4≥length(QE)≥3
	en=kikf2E(ki,kf)
	T=promote_type(eltype(QE),typeof(en))
	length(QE)<4 && (QE=vcat(convert.(T,QE),convert(T,en)))
	@assert QE[4]==en "Inconsistent Q-E 4-vector and ki,kf specification"
	# We can't *create* a variable inside of a try/catch block but the last value 
	# *will* fall out, which then gets returned.
	try
		angles=q2a(lengths,alpha,va,vb,QE;kfix=kf,sens=sens)
		Qxy=a2q(ki,angles[[3,4]]...,kf)
	catch
		Qxy=(NaN,NaN)
	end
end
function q2xy(lengths::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Matrix,ki,kf,sens=[-1,1,-1])
	nQ=size(QE,2)
	Qx=Array{Float64}(undef,nQ)
	Qy=Array{Float64}(undef,nQ)
	for i=1:nQ
		(Qx[i],Qy[i])=q2xy(lengths,alpha,va,vb,QE[:,i],ki,kf,sens)
	end
	(Qx,Qy)
end
function drawQEPoints(a::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Matrix,ki,kf,sens=[-1,1,-1];color="k",marker=".")
	(Qx,Qy)=q2xy(a,alpha,va,vb,QE,ki,kf,sens)
	plot(Qx,Qy,marker=marker,color=color, linestyle="none")
end


function drawTASPMA7(lengths::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Vector,ki,kf;
                     sens=[-1,1,-1],
	             dist=[80,150,120,110], # typical distances at TASP
                     sample_space_radius = 5/2, # MA7 sample space is ~50mm
		     magnet_outer_radius = 65.5/2, # MA7 outer diameter/2
                     mono_width=15, # monochromator width of TASP in cm
                     anal_width=17.5, # cm
                     det_width=5, #cm (max)
                     H::Vector=[0,0,0],
                     del0=NaN)
	if norm(H)>0
		ang_H=q2a(lengths,alpha,va,vb,H,kfix=kf)
		phi=sens[2]*(180-abs(ang_H[4]))/2
		del=phi+ang_H[3] # the sample-angle difference between va and H (assuming H is in the scattering plane)
		isnan(del0) && (del0=del)
		del0==del || warn("Specified magnet angle ($del0) and calculated magnet angle ($del) differ!")
	end

	angs=q2a(lengths,alpha,va,vb,QE,kfix=kf)

	# generate the *cumulative* angles to the Monochromator, Sample, Analyzer, and Detector
	ca=cumsum([0;angs[2:2:end]])
	bx=cumsum([0;dist.*cosd.(ca)]) # [guide-exit, monochromator, sample, analyzer, detector]
	by=cumsum([0;dist.*sind.(ca)])

	# draw the beam
	plot(bx,by)
	# draw the monochromator, analyzer, and detector
	plot(bx[2].+mono_width*[-0.5,0.5]*cosd(angs[1]),by[2].+mono_width*[-0.5,0.5]*sind(angs[1]))
	plot(bx[4].+anal_width*[-0.5,0.5]*cosd(angs[5]+ca[3]),by[4].+anal_width*[-0.5,0.5]*sind(angs[5]+ca[3]))
	drawpatch(arcxy(det_width/2,360;x0=bx[5],y0=by[5])...)
	# draw the magnet here
	ma=del0+(angs[3]+ca[2])
	nxy=magnet_outer_radius*-[cosd(ma),sind(ma)]
	for (θ0,color) in zip( (45,135,225,315), (0.4*[1,0,0],0.6*[0,1,0],0.6*[0,0,1],0.4*[1,1,1]) )
	drawpatch(vcat.(arcxy(magnet_outer_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma),reverse.(arcxy(sample_space_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma)) )...,color=color)
	end
	arrow(bx[3],by[3],nxy[1],nxy[2],fc="r",ec="r")

	PyPlot.axis("equal")
	xlabel("x [cm]")
	ylabel("y [cm]")
	nothing
end
function drawTASPMA02(lengths::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Vector,ki,kf;
                     sens=[-1,1,-1],
	             dist=[80,150,120,110], # typical distances at TASP
                     sample_space_radius = 5/2, # MA7 sample space is ~50mm
		     magnet_outer_radius = 65.5/2, # MA7 outer diameter/2
                     mono_width=15, # monochromator width of TASP in cm
                     anal_width=17.5, # cm
                     det_width=5, #cm (max)
                     H::Vector=[0,0,0],
                     del0=NaN)
	if norm(H)>0
		ang_H=q2a(lengths,alpha,va,vb,H,kfix=kf)
		phi=sens[2]*(180-abs(ang_H[4]))/2
		del=phi+ang_H[3] # the sample-angle difference between va and H (assuming H is in the scattering plane)
		isnan(del0) && (del0=del)
		del0==del || warn("Specified magnet angle ($del0) and calculated magnet angle ($del) differ!")
	end

	angs=q2a(lengths,alpha,va,vb,QE,kfix=kf)

	# generate the *cumulative* angles to the Monochromator, Sample, Analyzer, and Detector
	ca=cumsum([0;angs[2:2:end]])
	bx=cumsum([0;dist.*cosd.(ca)]) # [guide-exit, monochromator, sample, analyzer, detector]
	by=cumsum([0;dist.*sind.(ca)])

	# draw the beam
	plot(bx,by)
	# draw the monochromator, analyzer, and detector
	plot(bx[2].+mono_width*[-0.5,0.5]*cosd(angs[1]),by[2].+mono_width*[-0.5,0.5]*sind(angs[1]))
	plot(bx[4].+anal_width*[-0.5,0.5]*cosd(angs[5]+ca[3]),by[4].+anal_width*[-0.5,0.5]*sind(angs[5]+ca[3]))
	drawpatch(arcxy(det_width/2,360;x0=bx[5],y0=by[5])...)
	# draw the magnet here
	a3zero=(180+ca[2]) # at TASP, at least, a3-zero points towards the monochromator
	ma=(a3zero+angs[3])-del0
	nxy=magnet_outer_radius*[cosd(ma),sind(ma)]
	mdarkangs=MA02darkangles
	for i=1:size(mdarkangs,2)
		θ0=ma - mean(mdarkangs[:,i]) # the magnet needs to be rotated *by* darkangle for the direction along H to become blocked, so subtract the dark angle here
		Δθ= abs(mdarkangs[1,i]-mdarkangs[2,i])
		drawpatch( vcat.( arcxy(magnet_outer_radius,Δθ,x0=bx[3],y0=by[3],θ0=θ0),reverse.( arcxy(sample_space_radius,Δθ,x0=bx[3],y0=by[3],θ0=θ0) ) )..., color="black")
	end
	arrow(bx[3],by[3],nxy[1],nxy[2],fc="r",ec="r")

	PyPlot.axis("equal")
	xlabel("x [cm]")
	ylabel("y [cm]")
	title("Q=$(QE) H=$H")

	nothing
end

function drawCAMEAMA7(lengths::Vector,alpha::Vector,va::Vector,vb::Vector,QE::Vector,kf=1.3541786537497906;
                     sens=[1,-1,1],
	                   dist=[800,1500], # typical distances at RITA-II
                     sample_analyzer_dist=[930,994,1056,1120,1183,1247,1312,1379],
                     analyzer_widths=[72.0,81.8,91.6,102.4,112.2,119.1,128.1,139.2],
                     sample_space_radius = 50/2, # MA7 sample space is ~50mm
                     magnet_outer_radius = 655/2, # MA7 outer diameter/2
                     mono_width=150, # monochromator width of TASP in mm
                     H::Vector=[0,0,0],
                     del0=NaN)
	if norm(H)>0
		ang_H=q2a(lengths,alpha,va,vb,H,kfix=kf)
		phi=sens[2]*(180-abs(ang_H[4]))/2
		del=phi+ang_H[3] # the sample-angle difference between va and H (assuming H is in the scattering plane)
		isnan(del0) && (del0=del)
		del0==del || warn("Specified magnet angle ($del0) and calculated magnet angle ($del) differ!")
	end

	angs=q2a(lengths,alpha,va,vb,QE,kfix=kf,sens=sens)[1:4]
  angs=repeat(angs,outer=[1,8])
  angs[4,:]=angs[4,1]+sens[2]*7.5*(0:7) # the scattering angles for the 8 analyzer segments

  # the distances to each of the 8 analyzers per segment
  dists= reshape(vcat(repeat(dist,outer=[1,8]),sample_analyzer_dist'),(3,1,8))

	# generate the *cumulative* angles to the Monochromator, Sample, Analyzer
  ca=cumsum(vcat(zeros(1,8),angs[2:2:4,:]),1)
	bx=cumsum(vcat(zeros(1,8,8),dists.*cosd.(ca)),1) # [guide-exit, monochromator, sample, analyzer]
	by=cumsum(vcat(zeros(1,8,8),dists.*sind.(ca)),1)

	# draw the beam
	plot(bx[1:3,1,1],by[1:3,1,1]) # up to the sample
	# draw the monochromator, analyzer, and detector
	plot(bx[2,1,1]+mono_width*[-0.5,0.5]*cosd(angs[1,1]),by[2,1,1]+mono_width*[-0.5,0.5]*sind(angs[1,1]))
  for i=1:8,j=1:8
	plot(bx[4,i,j]+analyzer_widths[j]*[-0.5,0.5]*cosd(90+ca[3,i]),by[4,i,j]+analyzer_widths[j]*[-0.5,0.5]*sind(90+ca[3,i]))
  end
	# draw the magnet here
	ma=del0+(angs[3]+ca[2])
	nxy=magnet_outer_radius*-[cosd(ma),sind(ma)]
	for (θ0,color) in zip( mean(MAD.MA7darkangles,dims=1), [0.4*[1,0,0],0.6*[0,1,0],0.6*[0,0,1],0.4*[1,1,1]] )
	drawpatch(vcat.(arcxy(magnet_outer_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma),reverse.(arcxy(sample_space_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma)) )...,color=color)
	end
	arrow(bx[3],by[3],nxy[1],nxy[2],fc="r",ec="r")

	PyPlot.axis("equal")
	xlabel("x [mm]")
	ylabel("y [mm]")
	nothing
end
function drawCAMEAMA7(a1::Number,a2::Number,a3::Number,a4::Number;
                     sens=[1,-1],
	                   dist=[800,1500], # typical distances at RITA-II
                     sample_analyzer_dist=[930,994,1056,1120,1183,1247,1312,1379],
                     analyzer_widths=[72.0,81.8,91.6,102.4,112.2,119.1,128.1,139.2],
                     sample_space_radius = 50/2, # MA7 sample space is ~50mm
                     magnet_outer_radius = 655/2, # MA7 outer diameter/2
                     mono_width=150, # monochromator width of TASP in mm
                     H::Vector=[0,0,0],
                     del0=NaN)
	if norm(H)>0
		ang_H=q2a(lengths,alpha,va,vb,H,kfix=kf)
		phi=sens[2]*(180-abs(ang_H[4]))/2
		del=phi+ang_H[3] # the sample-angle difference between va and H (assuming H is in the scattering plane)
		isnan(del0) && (del0=del)
		del0==del || warn("Specified magnet angle ($del0) and calculated magnet angle ($del) differ!")
	end

  angs=abs.([a1,a2,a3,a4])
  angs.*=sens[[1,1,2,2]]
  angs=repeat(angs,outer=[1,8])
  angs[4,:]=angs[4,1]+sens[2]*7.5*(0:7) # the scattering angles for the 8 analyzer segments

  # the distances to each of the 8 analyzers per segment
  dists= reshape(vcat(repeat(dist,outer=[1,8]),sample_analyzer_dist'),(3,1,8))

	# generate the *cumulative* angles to the Monochromator, Sample, Analyzer
  ca=cumsum(vcat(zeros(1,8),angs[2:2:4,:]),1)
	bx=cumsum(vcat(zeros(1,8,8),dists.*cosd.(ca)),1) # [guide-exit, monochromator, sample, analyzer]
	by=cumsum(vcat(zeros(1,8,8),dists.*sind.(ca)),1)

	# draw the beam
	plot(bx[1:3,1,1],by[1:3,1,1]) # up to the sample
	# draw the monochromator, analyzer, and detector
	plot(bx[2,1,1]+mono_width*[-0.5,0.5]*cosd(angs[1,1]),by[2,1,1]+mono_width*[-0.5,0.5]*sind(angs[1,1]))
  for i=1:8,j=1:8
	plot(bx[4,i,j]+analyzer_widths[j]*[-0.5,0.5]*cosd(90+ca[3,i]),by[4,i,j]+analyzer_widths[j]*[-0.5,0.5]*sind(90+ca[3,i]))
  end
	# draw the magnet here
	ma=del0+(angs[3]+ca[2])
	nxy=magnet_outer_radius*-[cosd(ma),sind(ma)]
	for (θ0,color) in zip( mean(MAD.MA7darkangles,dims=1), [0.4*[1,0,0],0.6*[0,1,0],0.6*[0,0,1],0.4*[1,1,1]] )
	drawpatch(vcat.(arcxy(magnet_outer_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma),reverse.(arcxy(sample_space_radius,45;x0=bx[3],y0=by[3],θ0=θ0+ma)) )...,color=color)
	end
	arrow(bx[3],by[3],nxy[1],nxy[2],fc="r",ec="r")

	PyPlot.axis("equal")
	xlabel("x [mm]")
	ylabel("y [mm]")
	nothing
end

function drawPath(path::Vector{NTuple{2,T}};k...) where T
	xy=points2vectors(path)
	h=drawpatch(xy;k...)
	xyl=extrema.(xy)
	xlim(extrema([xlim()...,xyl[1]...]))
	ylim(extrema([ylim()...,xyl[2]...]))
	return h
end
function drawPath(path::Vector{Vector{T}};k...) where T
	xy=points2vectors(path)
	h=drawpatch(xy;k...)
	xyl=extrema.(xy)
	xlim(extrema([xlim()...,xyl[1]...]))
	ylim(extrema([ylim()...,xyl[2]...]))
	return h
end


function darkanglepath(θs::Vector{T},n=0) where T<:AbstractFloat
  c=T(180)
  α,β=extrema(θs)
  [[α,-c],[β,-c],[β,c],[α,c],[α-2c,-c],[β-2c,-c],[α,c-β+α]] .+ [[2c*T(n),0]]
end
darkanglepath(θs::Vector{T},n=0) where T<:Number = darkanglepath(convert.(Float64,θs),n)

function experimentwindows(darkangles::Vector{Vector{Vector{T}}},experimentrange::Vector{Vector{R}}) where {T,R}
	(success,experimentdarkangles)=pathintersection(darkangles,experimentrange)
	success || error("Something has gone wrong with easyClipper.pathintersection")
	(success,windows)=pathxor(experimentdarkangles,experimentrange)
	success || error("Something has gone wrong with easyClipper.pathxor")
	return windows
end
function experimentwindows(darkangles::Matrix{T};a3min=0,a3max=360,a4min=0,a4max=180,minarea=1) where T<:AbstractFloat
	@assert size(darkangles,1)==2 "Dark angles should be specified as columns of the passed 2xN matrix"
  exprange=Vector{T}[T[a3min,a4min],T[a3max,a4min],T[a3max,a4max],T[a3min,a4max]]
	# each dark angle set will produce a track through (a3,a4) space from a3_min-360 to a3_max
	# try to be clever about which dark angle paths we need
	n1= cld(maximum(darkangles)-min(a3min,a3max),360)+1
	n2= cld(minimum(darkangles)-max(a3min,a3max),360)+1
	n=max(n1,n2)
	daps=vec( [darkanglepath(darkangles[:,i],j) for i=1:size(darkangles,2), j=-n:n ] )
	ew=experimentwindows(daps,exprange)
  ew[ area.(ew).>minarea ] # rounding errors in easyClipper can return "accessible" windows with ~0 area
end

# magnet dark-angles defined relative to the horizontal field direction
# and such that rotating a3 to between the specified angles blocks the incident beam
const MA7darkangles=hcat([ [22.5,22.5+45].+90x for x in 0:3]...)
const MA02darkangles=hcat(30 .+ 12/2*[-1,1],-180-40 .+ 6/2*[-1,1])
const MA11darkangles=hcat([17/2,90-8/2],[90+8/2,180-17/2],[180+17/2,270-8/2],[270+8/2,360-17/2])

const MA7a2limits=[41,93.]
const MA02a2limits=[-Inf,Inf]
const MA11a2limits=[-Inf,Inf]

function get_magnet_darkangles_limits(magnet::Symbol)
  if magnet == :MA7
    return (MA7darkangles,MA7a2limits)
  elseif magnet==:MA02
    return (MA02darkangles,MA02a2limits)
  elseif magnet==:MA11
    return (MA11darkangles,MA11a2limits)
  else
    error("Unknown magnet $magnet")
  end
end

function drawMaga3a4(magnet::Symbol,del0=0,N=100;
                    accessible="white",inaccessible="grey",alpha=1,a3min=-Inf,a3max=+Inf,a4max=119,a4min=0)
  #
  (da,a2l)=get_magnet_darkangles_limits(magnet)
  isfinite(a3min) || (a3min=minimum(mean(da,dims=1)))
  isfinite(a3max) || (a3max=a3min+360)
  gca()[:set_facecolor](inaccessible)
  # draw the physically accessible Q space for this ki,kf (which isn't blocked by the magnet)
  notblocked = points2vectors.([outlinepath(x,N) for x in experimentwindows(da.+del0;a3min=a3min,a3max=a3max,a4min=a4min,a4max=a4max)])
  magacc = [drawpatch( a3a4, color=accessible, alpha=alpha ) for a3a4 in notblocked];

  PyPlot.axis("equal")
  legend([magacc[1]],["accessible"])
  xlabel("a3 [°]"),ylabel("a4 [°]"),
  return magacc[1]
end
function drawMagQxy(magnet::Symbol,ki::Number,kf::Number,del0=0,N=100;
                    accessible="white",inaccessible="grey",alpha=1,a4max=119,a4min=0)
  #
  (da,a2l)=get_magnet_darkangles_limits(magnet)
  a3min=minimum(mean(da,dims=1)); a3max=a3min+365
  a2=k2a2(ki)
  if a2<minimum(a2l)||a2>maximum(a2l)
    warn("The passed incident wavevector is outside of the allowed full-field range for $magnet")
    accessible="red"
  end
  # in general, we can't access most of Qx,Qy space. so color the whole axes
  gca()[:set_facecolor](inaccessible)
  # draw the physically accessible Q space for this ki,kf (which isn't blocked by the magnet)
  notblocked = points2vectors.([outlinepath(x,N) for x in experimentwindows(da.+del0;a3min=a3min,a3max=a3max,a4min=a4min,a4max=a4max)])
  magacc = [drawpatch( a2q(ki,a3a4,kf), color=accessible, alpha=alpha ) for a3a4 in notblocked];

  PyPlot.axis("equal")
  legend([magacc[1]],["accessible"])
  xlabel("Q\$_x\$ [1/\$\\AA\$]"),ylabel("Q\$_y\$ [1/\$\\AA\$]"),
  gca()[:set_title]("E="*labfmt(kikf2E(ki,kf))*" meV \$k_i="*labfmt(ki)*"\\AA^{-1}\$ \$k_f="*labfmt(kf)*"\\AA^{-1}\$")
  return magacc[1]
end

function fancy_int(a::Integer,A::AbstractString)
  isneg=signbit(a); a=abs(a)
  (isneg ? "-" : "") * (a==0 ? "0" : a==1 ? A : "$a"*A)
end
function fancy_int(a::Real,A::AbstractString)
  isneg=signbit(a); a=abs(a)
  (isneg ? "-" : "") * (isapprox(a,0) ? "0" : isapprox(a,1) ? A : "$a"*A)
end

function fancy_hkl(h::Real,k::Real,l::Real;H="η",K="κ",L="λ")
  fh=fancy_int(h,H)
  fk=abs(k)==abs(h) ? fancy_int(k,H) : fancy_int(k,K)
  fl=abs(l)==abs(h) ? fancy_int(l,H) : abs(l)==abs(k) ? fancy_int(l,K) : fancy_int(l,L)
  length(fh)==length(fk)==length(fl)==1 ? fh*fk*fl : fh*" "*fk*" "*fl
end
				

function drawMagQhkl(magnet::Symbol,lat::Vector,ang::Vector,v1::Vector,v2::Vector,
                     ki::Number,kf::Number,del0=0,N=100;
                     accessible="white",inaccessible="grey",alpha=1,a4max=119,a4min=0)
  #
  (da,a2l)=get_magnet_darkangles_limits(magnet)
  a3min=minimum(mean(da,dims=1)); a3max=a3min+360
  a2=k2a2(ki)
  if a2<minimum(a2l)||a2>maximum(a2l)
    warn("The passed incident wavevector is outside of the allowed full-field range for $magnet")
    accessible="red"
  end
  lx=norm([q2xy(lat,ang,v1,v2,v1,ki,kf)...])
  ly=norm([q2xy(lat,ang,v1,v2,v2,ki,kf)...])
  # in general, we can't access most of Qx,Qy space. so color the whole axes
  gca()[:set_facecolor](inaccessible)
  # draw the physically accessible Q space for this ki,kf (which isn't blocked by the magnet)
  notblocked = points2vectors.([outlinepath(x,N) for x in experimentwindows(da.+del0;a3min=a3min,a3max=a3max,a4min=a4min,a4max=a4max)])
  magacc = [drawpatch( a2q(ki,a3a4,kf)./(lx,ly), color=accessible, alpha=alpha ) for a3a4 in notblocked];

  PyPlot.axis("equal")
  legend([magacc[1]],["accessible"])
  xlabel("($(fancy_hkl(v1...))) [rlu]")
  ylabel("($(fancy_hkl(v2...))) [rlu]")
  
  gca()[:set_title]("E="*labfmt(kikf2E(ki,kf))*" meV \$k_i="*labfmt(ki)*"\\AA^{-1}\$ \$k_f="*labfmt(kf)*"\\AA^{-1}\$")
  return magacc[1]
end

const MA7=:MA7
const MA02=:MA02
const MA11=:MA11

for (ext,fnc) in zip( (:a3a4,:Qxy,:Qhkl),(:drawMaga3a4,:drawMagQxy,:drawMagQhkl) )
for m in (:MA7,:MA02,:MA11)
  f=Symbol("draw",m,ext)
  @eval $f(o...;k...)=$fnc($m,o...;k...)
end
end

end # Module MAD
