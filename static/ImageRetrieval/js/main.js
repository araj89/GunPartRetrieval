var StartProgress = 0;
var ProgressTimeInterval = 1200;
var FilterValue = 1;

var g_s = 0;
var g_n = 0;
var g_scores = new Array();
var g_scores2 = new Array();
var g_files = new Array();

IMG_NOT_EXIST = -101;
DIRECTORY_NOT_EXIST = -102;
CAPTURE_FAILED = -103;
IMG_PROC_ERROR = -104;
DATABASE_ERROR = -110;
RES_FOLDER_ERROR = -111;
SAVE_OK = 0;

function move() {
  if (StartProgress == 0) {
    StartProgress = 1;
    var elem = document.getElementById("myBar");
    var width = 1;
    var id = setInterval(frame, ProgressTimeInterval);
    function frame() {
        if (StartProgress == 1) {
            $.ajax({
                type: "POST",
                url: '/progress',
                success: function (data) {
                    width = data;

                    if (width >= 100) {
                        elem.style.width = 100 + "%";
                        clearInterval(id);
                        StartProgress = 0;
                    } else {
                        elem.style.width = width + "%";
                    }
                }
            });
        }
    }
  }
}

function slider_filter_val(strv) {
    v = parseInt(strv);
    //map [1, 100] -> [100.0, 1.0 : 0.1]
    if (v < 10)
        FilterValue = 100.0 * (10 - v);
    else if (v < 90)
        FilterValue = 1.0 - 0.01 * v;
    else
        FilterValue = 0.01 * (100 - v);
}
function Camera() {
    bCapture = false;
    var slider_filter = document.getElementById("res_filter");
    slider_filter.value = '1';
    slider_filter_val('1');
    $('#camera_btn').attr("disabled", '');
    $('#capture_btn').attr("disabled", null);
    $.ajax({
        type: "POST",
        url: '/camera',
        success: function(data){
            console.log('camera called');
        }
    });
}

function Capture() {
    bCapture = true;
    var slider_filter = document.getElementById("res_filter");
    slider_filter.value = '1';
    slider_filter_val('1');

    $('#capture_btn').attr("disabled", '');
    $('#camera_btn').attr("disabled", null);
    $.ajax({
        type: "POST",
        url: '/capture',
        success: function(data){
            console.log('capture called');
        }
    });
}

function tmpl_save() {
    filename = $('#tmpl-title').val();
    filename = 'static/ImageRetrieval/tmpls/' + filename;

    var search_method = $("input[type=radio]:checked").val();
    var min_rate = $("#min_rate").val();
    var nres = $("#nres").val();
    data = {
        'filename' : filename,
        'search_method' : search_method,
        'min_rate' : min_rate,
        'nres' : nres
    };

    $.ajax({
       type : "POST",
       url : "/set_file_info",
       data : data,
       success : function (data) {
            alert("Successfully saved!");
       } 
    });
}

function tmpl_load() {
    filename = $('#tmpl-title').val();
    filename = 'static/ImageRetrieval/tmpls/' + filename;

    $.ajax({
       type : "POST",
       url : "/get_file_info",
       data : {"name" : filename},
       success : function (data) {
           jdata = JSON.parse(data);

           search_method = jdata['search_method'];
           scale_range_int = jdata['scale_range'];
           res_number = jdata['res_number'];

           if (search_method == 0) radio = document.getElementById("method0");
           else if (search_method == 1) radio = document.getElementById("method1");
           else radio = document.getElementById("method2");
           radio.checked = true;

           var slider = document.getElementById('min_rate');
           slider.value = scale_range_int.toString();
           var output = document.getElementById("rate_val");
           output.innerHTML = slider.value;

           $('#nres').val(res_number);
       }
    });
}
function set_tmpl(file) {
    $('#modal_ok').trigger('click');
    $('#tmpl-title').val(file);
    
    // read file content using XMLHttpRequest---------------
   /*var rawFile = new XMLHttpRequest();
   rawFile.open("GET", fpath+file, false);
   rawFile.onreadystatechange = function () {
       if (rawFile.readyState === 4) {
           if (rawFile.status === 200 || rawFile.status == 0) {
               var allText = rawFile.responseText;
               alert(allText);
           }
       }
   };
   rawFile.send(null);*/

}

function tmpl_open() {
    $.ajax({
        type: "POST",
        url: '/get_tmpl_names',
        success: function(data){
            jdata = JSON.parse(data);
            names = jdata['list'];

            htm = '';
            for (i = 0; i < names.length; i++){
                file = names[i];
                htm += '<div style="width: 80%; text-align: center; margin-left: 10%;" onclick="set_tmpl(\'' + file + '\')"><h1 class = "imgHoverClass">' + file + '</h1></div>';
            }

            htm = '<div style="text-align: center;">' + htm + '</div>';

            $("#modal-content2").html(htm);
        }
    });
}

function ShowOrigImage(fpath, k, score, fname) {
    //alert(fpath + fname);
    htm = '<div><span>' + fpath + fname + '   :   ' + score + '</span><img src = "static/ImageRetrieval/assets/' + k + '.jpg"></div>';
    $("#modal-content2").html(htm);
}
function MakeScripts(s, n, scores, scores2, files) {

    /*
    <div class="imgHoverClass col-md-4" onclick="javascript:alert(2);">
                        <img src="static/ImageRetrieval/assets/1.jpg" width="100%">
                        <div class="imgNamClass">
                            <span>2</span>
                        </div>
                    </div>
    */

    res = '';
    for (i = 0; i < n; i++) {
        score2 = scores2[i];
        if (score2 > FilterValue)
            continue;

        score = scores[i];
        file = files[i];
        fpath = file[0];
        fname = file[1];

        k = s + i;
        i1 = i + 1;
        res += '<div class="imgHoverClass col-md-4" data-target="#userRegisterModal" data-toggle="modal" ' +
            'onclick="ShowOrigImage(\'' + fpath +'\',\'' + k +'\',\'' + score +'\',\'' + fname + '\');">  ' +
            '<img src="static/ImageRetrieval/assets/' + k + '.jpg" width="100%">' +
            '<div class = "imgNamClass">' +
            '<span>' + i1 + ' : ' + fname +'</span>' +
            '</div></div>';
    }

    return res;
}

function checkValue(value,genre) {
	if(value == null || value=="") return false;

	var re;
	if(genre=="date")
		re=new RegExp(/^((19|20)\d{2})[\-\/](1[0-2]|0?[1-9])[\-\/](0?[1-9]|[12][0-9]|3[01])$/);
	else if(genre=="datetime")
		re=new RegExp(/^((19|20)\d{2})[\-\/](1[0-2]|0?[1-9])[\-\/](0?[1-9]|[12]\d|3[01])\s(2[0-3]|[01]?\d):([0-5]?\d):([0-5]?\d)$/);
	else if(genre=="time")
		re=new RegExp(/^(2[0-3]|[01]?\d):([0-5]?\d):([0-5]?\d)$/);
	else if(genre=="timehm")
		re=new RegExp(/^(2[0-3]|[01]?\d):([0-5]?\d)$/);
	else if(genre=="number")
		re=new RegExp(/^\d+$/);
	else if(genre=="float" || genre=="pfloat" || genre=="pzfloat")
		re=new RegExp(/^\-?\d+(\.?\d+)?$/);
	else if(genre=="hexadecimal")
		re=new RegExp(/^([0-9A-F])+$/);
	else if(genre=="phone")
		re=new RegExp(/^(\d)?\-?(\d{2})?\-?(\d{3})?\-?\d{4}/);
	else if(genre=="email")
		re=new RegExp(/^[A-Za-z]+[\w\-]*@[A-Za-z]+[\w\-]*(\.[A-Za-z]+[\w\-]*)+$/);
	else if(genre=="domain")	//original code re=new RegExp(/^[A-Za-z]+\w*(\.[A-Za-z]+\w*)+$/);
		re=new RegExp(/^\w+(\.\w+)+$/);
	else if(genre=="id")
		re=new RegExp(/^[_\-\w]*$/);
	else if(genre=="ip")
		re=new RegExp(/^([0-9]{1,2}|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.([0-9]{1,2}|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.([0-9]{1,2}|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.([0-9]{1,2}|1[0-9]{2}|2[0-4][0-9]|25[0-5])$/);
	else if(genre=="url")
		re=new RegExp(/^http:\/\/[A-Za-z0-9]+\w*([\.:]{1}[A-Za-z0-9]+\w*)+$/);
	else if(genre=="content")
		re=new RegExp(/^[^\s]+$/);
	else if(genre=="space")
		re=new RegExp(/^\s+$/);
	else if(genre=="dialplan")
		re=new RegExp(/^([NZX\d]*|(\[([NZX\d]|(\d\-\d))+\])*)+\.?$/);
	else if(genre=="mac")
		re=new RegExp(/^([\dA-Fa-f]{2}:){5}[\dA-Fa-f]{2}$/);
	else if(genre=="korean"){
		for (i=0; i < value.length ; i ++) {
			if (value.charCodeAt(i) < 0xAC00 || value.charCodeAt(i) > 0xD7A3)
				return false;
		}
		return true;
	}
	else return false;

	if(value.search(re)!=-1){
		if (genre == "pfloat" && value <= 0) return false;
		else if (genre == "pzfloat" && value < 0) return false;
		return true;
	}

	return false;
}

function fieldCheck(save_repo, search_method, min_rate, nres) {
    if (save_repo == '') {
        alert("Please set the save directory");
        return -1;
    }

    if (search_method != 0 && search_method != 1 && search_method != 2) {
        alert ("programming error2");
        return -1;
    }

    if (!checkValue(min_rate, "number")){
        alert ("Scale range must be number");
        return -1;
    }
    if (min_rate < 1 || min_rate > 100) {
        alert("Scale range out error");
        return -1;
    }

    if (!checkValue(nres, "number")) {
        alert ("Result numbers must be number type");
        return -1;
    }

    return 0;
}

function Search() {
    if (bCapture == false) {
        alert('Please capture the image to search');
        return;
    }

    var slider_filter = document.getElementById("res_filter");
    slider_filter.value = '1';
    slider_filter_val('1');

    $('#result-text').html('<h3>Searching the images from the database</h3>');
    $('#search_btn').attr("disabled", '');

    g_s = 0;
    g_n = 0;
    g_scores = new Array();
    g_scores2 = new Array();
    g_files = new Array();

    //var save_dir = $("#save-repo").val();
    var search_method = $("input[type=radio]:checked").val();
    var min_rate = $("#min_rate").val();
    var nres = $("#nres").val();

    if (fieldCheck("ddd", search_method, min_rate, nres) != 0) {
        $('#search_btn').attr("disabled", null);
        return;
    }

    // min_rate range change [1, 100] => [1.05, 1.5] (delta = 0.5 / 100 = 0.005)
    min_rate = 1.0 + 0.005 * min_rate;

    //--- search engine called-----------
    move();
    $("#searched-imgs").html('');
    data = {
        "search_method" : search_method,
        "min_rate" : min_rate,
        "nres" : nres
    };

    $.ajax({
        type: "POST",
        url: '/search',
        data : {
            "data": JSON.stringify(data)
        },
        success: function(data){
            jdata = JSON.parse(data);

            n = jdata['n'];
            s = jdata['s'];
            scores = jdata['scores'];
            scores2 = jdata['scores2'];
            files = jdata['files'];
            res_str = jdata['res_str'];

            if (n == IMG_NOT_EXIST) {
                StartProgress = 0;
                $('#search_btn').attr("disabled", null);
                alert("Please connect camera and capture image");
            }
            else if (n == IMG_PROC_ERROR) {
                StartProgress = 0;
                $('#search_btn').attr("disabled", null);
                alert("Please capture the right image");
            }
            else if (n == RES_FOLDER_ERROR) {
                StartProgress = 0;
                $('#search_btn').attr("disabled", null);
                alert("Please type the Save directory correctly");
            }
            else {
                g_s = s;
                g_n = n;
                g_scores = scores;
                g_scores2 = scores2;
                g_files = files;

                str_htm = MakeScripts(s, n, scores, scores2, files);
                $("#searched-imgs").html(str_htm);
                $('#search_btn').attr("disabled", null);
                $('#result-text').html(res_str);
            }
        }
    });

}

function StoreDirectoryToDataset() {
    $('#store_directory_btn').attr("disabled", '');
    $('#result-text').html('<h3>Converting the images to database</h3>');

    var search_dir1 = $("#image-repo1").val();
    var search_dir2 = $("#image-repo2").val();
    var search_dir3 = $("#image-repo3").val();
    var search_repos = [search_dir1, search_dir2, search_dir3];

    var check1 = document.getElementById('check-repo1');
    var check_dir1 = 0;
    if (check1.checked) check_dir1 = 1;

    var check2 = document.getElementById('check-repo2');
    var check_dir2 = 0;
    if (check2.checked) check_dir2 = 1;

    var check3 = document.getElementById('check-repo3');
    var check_dir3 = 0;
    if (check3.checked) check_dir3 = 1;

    var search_checks = [check_dir1, check_dir2, check_dir3];

    if (search_repos.length != 3 || search_checks.length != 3) {
        $('#store_directory_btn').attr("disabled", null);
        alert("programming error1");
        return -1;
    }

    flag = -1;
    for (i = 0;i < 3; i++) {
        s_repo = search_repos[i];
        s_check = search_checks[i];
        if (s_repo != '' && s_check == 1)
            flag = 1;
    }
    if (flag == -1) {
        $('#store_directory_btn').attr("disabled", null);
        alert("Please set the search direcitory correctly");
        return -1;
    }

    ncnt = 0;
    var ne_dirs = new Array();
    for (i = 0; i < 3; i++) {
        d = search_repos[i];
        c = search_checks[i];
        if (c == 1 && d != '')
            ne_dirs[ncnt++] = d;
    }

    move();

    data = {
        "search_path" : ne_dirs
    };

    $.ajax({
        type: "POST",
        url: '/convert_directories',
        data : {
            "data": JSON.stringify(data)
        },
        success: function(data) {
            $('#store_directory_btn').attr("disabled", null);
            jdata = JSON.parse(data);
            res_code = jdata['res_code'];
            str_res = jdata['str_res'];

            if (res_code != SAVE_OK)
                alert(str_res);
            else {
                $('#result-text').html(str_res);
            }
        }
    });

}

function StoreCapImageToDataset() {
    if (bCapture == false) {
        alert('Please capture the image to store');
        return;
    }

    fname = $('#capture-name').val();
    //fpath = $('#capture-path').val();
    fpath = 'ddd';
    if (fname == '') {
        alert('Please enter the SKU field');
        return;
    }
    if (fpath == '') {
        alert('Please enter the path field');
        return;
    }

    data = {'fname' : fname, 'fpath' : fpath};
    $.ajax({
        type: "POST",
        url: '/save_captured_img',
        data : data,
        success: function(data){
            alert(data);
        }
    });

}

