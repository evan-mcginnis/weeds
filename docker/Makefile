#
# M A K E F I L E
#
clean:
	rm -rf jetson lib 
	#docker rmi -f $(docker images -aq)

container:
	cp -r ../jetson .
	cp -r ../lib . 
	docker build -t emcginnis/parameters .

container-tag:
	docker tag parameters:latest parameters:latest

# Super-bad form to put the password here, but we want this to go to my account
docker-push:
	docker login -u emcginnis -p AZj7X7jyzf9W
	docker push emcginnis/parameters

#docker tag local-image:tagname new-repo:tagname
#docker push new-repo:tagname
