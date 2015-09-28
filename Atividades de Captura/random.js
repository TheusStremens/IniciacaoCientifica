function sortearAtvidades() {
	$(function(){
    $('#portfolioModal3').on('show.bs.modal', function(){
        var myModal = $(this);
        clearTimeout(myModal.data('hideInterval'));
        myModal.data('hideInterval', setTimeout(function(){
            myModal.modal('hide');
            var x = document.getElementsByClassName("portfolio-item");
            for(var i = 0; i < x.length; i++) {
				if(x[i].getElementsByTagName("a")[0].href.split("#")[1].toString() == "portfolioModal3" )
					x[i].setAttribute("style", "opacity: 0.5; pointer-events: none; cursor: default;");
			}
        }, 180000));
    });
	});
	$(function(){
    $('#portfolioModal4').on('show.bs.modal', function(){
        var myModal = $(this);
        clearTimeout(myModal.data('hideInterval'));
        myModal.data('hideInterval', setTimeout(function(){
            myModal.modal('hide');
            var x = document.getElementsByClassName("portfolio-item");
            for(var i = 0; i < x.length; i++) {
				if(x[i].getElementsByTagName("a")[0].href.split("#")[1].toString() == "portfolioModal4" )
					x[i].setAttribute("style", "opacity: 0.5; pointer-events: none; cursor: default;");
			}
        }, 180000));
    });
	});
    //alert("I am an alert box!");
    var arr = [];
	while(arr.length < 3){
		var randomnumber=Math.floor(Math.random()*11 + 1);
		var found=false;
		for(var i=0;i<arr.length;i++){
			if(arr[i]==randomnumber){
				found=true;
				break;
			}
		}
		if(!found)
			arr[arr.length]=randomnumber;
		}
	var portfolio = "#portfolioModal"
	var x = arr[0].toString();
	var dest = portfolio.concat(x);
    document.getElementById("atividade1").href=dest; 
    
    x = arr[1].toString();
    dest = portfolio.concat(x);
    document.getElementById("atividade2").href=dest; 

    x = arr[2].toString();
    dest = portfolio.concat(x);
    document.getElementById("atividade3").href=dest; 
}

function desativarAtividade(portfolioModal1) {
	var x = document.getElementsByClassName("portfolio-item");
	//console.log(x);
	for(var i = 0; i < x.length; i++) {
		if(x[i].getElementsByTagName("a")[0].href.split("#")[1].toString() == portfolioModal1.toString() )
			x[i].setAttribute("style", "opacity: 0.5; pointer-events: none; cursor: default;");
	}
}

