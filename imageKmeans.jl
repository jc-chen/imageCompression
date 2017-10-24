include("clustering2Dplot.jl")

type PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
end

# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	assert(d==d2)
	return X1.^2*ones(d,t) + ones(n,d)*(X2').^2 - 2X1*X2'
end

function kMeans(X0,k;doPlot=false)
	(nr,nc,d) = size(X0)
	n=nr*nc
	X = reshape(X0,n,d)

	# Choose random points to initialize means
	W = zeros(k,d)
	perm = randperm(n)
	for c = 1:k
		W[c,:] = X[perm[c],:]
	end

	# Initialize cluster assignment vector
	y = zeros(Int64, n)
	changes = n

	while changes != 0

		# Compute (squared) Euclidean distance between each point and each mean
		D = distancesSquared(X,W)

		# Assign each data point to closest mean (track number of changes labels)
		changes = 0
		for i in 1:n
			(~,y_new) = findmin(D[i,:])
			changes += (y_new != y[i])
			y[i] = y_new
		end

		# Optionally visualize the algorithm steps
		if doPlot && d == 2
			clustering2Dplot(X,y,W)
			sleep(.1)
		end

		# Find mean of each cluster
		for c in 1:k
			W[c,:] = mean(X[y.==c,:],1)
		end

		# Optionally visualize the algorithm steps
		if doPlot && d == 2
			clustering2Dplot(X,y,W)
			sleep(.1)
		end

		@printf("Running k-means, changes = %d\n",changes)
	end

	function predict(Xhat0)
		(tr,tc,d) = size(Xhat0)
		t=tr*tc
		Xhat = reshape(Xhat0,t,d)
		D = distancesSquared(Xhat,W)

		yhat = zeros(Int64,t)
		for i in 1:t
			(~,yhat[i]) = findmin(D[i,:])
		end
		return yhat
	end

	return PartitionModel(predict,y,W)
end
