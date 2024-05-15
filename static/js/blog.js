
 
    getRandomId = function () {
        return 'id'.concat(Math.floor((Math.random() * 10000000000000000)).toString());
    };
    getCurrentTime = function () {
        return new Date().toLocaleString();
    };
 
    var curArr = null;
 
    var commentList = [];
 
    window.onload = function (ev) {
        // the function will be called at first while page loads
        initTreeComment();
        loadEmoji();
    };
 
    function loadEmoji() {
        const dom = domById('emoji_board');
        // &#128540;
        const fragment = document.createDocumentFragment();
        let num = 128512;
        for (let i = 0; i < 6; ++i) {
            const p = document.createElement('p');
            p.className = 'emoji_row';
            const pf = document.createDocumentFragment();
            for (let j = 0; j < 9; ++j) {
                const span = document.createElement('span');
                span.className = 'emoji_item'
                span.innerHTML = '&#' + num + ';';
                span.onclick = function () {
                    setEmojiToContent(this.innerText);
                }
                pf.append(span);
                ++num;
            }
            p.appendChild(pf);
            fragment.append(p);
        }
        dom.appendChild(fragment);
    }
 
    function setEmojiToContent(emo) {
        domById('comment_space').value += emo;
    }
 
    function emojiTrigger(ev) {
        ev.stopPropagation();
        const dom = domById('emoji_board');
        const block = dom.style.display;
        if (block === 'block') {
            dom.style.display = 'none';
        } else {
            dom.style.display = 'block';
        }
    }
 
    function login() {
        // const commentator = domById('commentator').value.trim();
		const commentator = "mnbvcxzz";
		console.log(commentator)
        if (commentator.length > 0) {
            domById('role_info_name').innerText = commentator;
        }
        initTreeComment();
    }
 
    /**
     * 初始化评论模块
     */
    initTreeComment = function () {
        const commentView = domById('comment_view');
        if (commentList.length === 0) {
            noComment(commentView)
        } else {
            haveComment(commentView, commentList)
        }
    };
 
    noComment = function (commentView) {
        commentView.style.textAlign = 'center';
        commentView.innerText = 'No comments'
    };
 
    /**
     * 点击 “提交” 按钮调用的函数
     * @param ev
     */
    function submitTree (ev) {
		login();
        const textareaEle = domById('comment_space');
        ev.stopPropagation();
        const text = textareaEle.value;
        const commentator = domById('role_info_name').innerText;
 
        if (text.length === 0 || commentator === '未登录') {
            alarmIfEmpSpace();
            return;
        }
 
        let plcHolder = textareaEle.placeholder;
 
        if(plcHolder.charAt(0) === '回') {
            plcHolder = plcHolder.substring(plcHolder.indexOf(' ') + 1, plcHolder.length)
        } else {
            curArr = commentList;
            plcHolder = "anonymous";
        }
 
        if (commentator === plcHolder) {
            cancelReply(event, domById('cancel_reply'));
            alert("你不能评论或回复自己!");
            return;
        }
 
        /*
        * hash表用于保存当前评论的点赞角色名称，默认值位 false
        * */
        const admirers = new Map();
        admirers.set(commentator, false);
        const commentTree = {
            id: getRandomId(),
            commentator: commentator,
            author: plcHolder,
            comment: text,
            time: getCurrentTime(),
            thumbUp: 0,
            admirers: admirers,
            reply: []
        };
        curArr.push(commentTree);
        initTreeComment();
        curArr = null;
        cancelReply(event, domById('cancel_reply'));
    }
 
    alarmIfEmpSpace = function () {
        alert("The comment content or name are empty！")
    };
 
    /**
     * 取消回复
     * @param ev 事件
     * @param th 回复按钮自身的 DOM 属性
     */
    function cancelReply (ev, th) {
        ev.stopPropagation();
        th.style.display = 'none';
        domById('comment_space').value = "";
        countTextareaLength();
        fillPlaceholderOfCommentSpace("请输入您要评论的内容");
    }
 
    /**
     * 计算输入框的内容长度
     */
    function countTextareaLength () {
        const space = domById('comment_space');
        document.getElementById('char_count').innerText = (1000 - space.value.length).toString();
    }
 
    /**
     * 输入框触发按键事件，并执行相关操作，比如记录输入内容的长度
     * @param ev 事件
     */
    function readKey (ev) {
        const space = domById('comment_space'), len = space.value.length;
        document.getElementById('char_count').innerText = (1000 - len).toString();
        handleCancelButton(space, len > 0 ? 'inline-block' : 'none')
    }
 
    function handleCancelButton (textArea, status) {
        domById('cancel_reply').style.display = status;
    }
 
    /**
     * 评论加载
     * @param commentView
     * @param arr
     */
    haveComment = function (commentView, arr) {
 
        commentView.style.textAlign = 'left';
 
        let htmlText = '';
        arr.forEach(function (value) {
            htmlText +=
                `<ul id="${value.id}" class="comment_ulist" style="">`+
                `  <li class="comment_line_box" id="${value.id.substring(0, 6)}">` +
                `    <div class="cmt_img">` +
                `      <img alt="头像" class="avatar" src="img/avt.jpg" width="30px" style="border-radius: 100%">` +
                `    </div>` +
                `    <div class="cmt_info">` +
                `      <a class="commentator">${value.commentator}</a>` +
                `      <span class="time">${value.time}</span>` +
                `   </div>` +
                `   <div class="cmt_reply">` +
                `       <span id="${value.id.substring(0, 9)}">` +
                `           <a class="cmt_reply_btn" id="${value.id.substring(0, 7)}">回复</a>` +
                `       </span>`;
 
            if(value.reply.length > 0) {
                htmlText += `<a id="${value.id.substring(0, 8)}" class="cmt_reply_checker">查看回复(${value.reply.length})</a>`
            }
            // 初始化当前角色是否给该条留言点赞了
            const commentator = domById('role_info_name').innerText;
            const isAdmirer = value.admirers.get(commentator);
            let color = 'rgba(0,0,0,0.9)';
            if (!isAdmirer) {
                color = 'rgba(0,0,0,0.4)';
            }
            htmlText += "" +
                `   <span>` +
                `       <a style="color:${color}" id="${value.id.substring(0, 10)}" class="thumb_up">${(value.thumbUp > 0 ? value.thumbUp : "")}</a>` +
                `   </span>` +
                `</div>`;
            htmlText += ` <div style="display: block; margin-top: 8px" class="comment">${value.comment}</div>`;
            htmlText += `</li></ul>`;
        });
 
        commentView.innerHTML = htmlText;
 
        showButton(arr, 1)
    };
 
    function thumbUp(value) {
        const dom = domById(value.id.substring(0, 10));
        dom.onclick = function (ev) {
            const cnt = value.thumbUp;
            const commentator = domById('role_info_name').innerText;
            const isAdmired = value.admirers.get(commentator);
            if (!value.admirers.has(commentator)) {
                value.admirers.set(commentator, false);
            }
            if (isAdmired) {
                value.thumbUp -= 1;
                dom.innerText = value.thumbUp === 0 ? '' : (cnt - 1).toString();
                dom.style.color = 'rgba(0,0,0,0.4)';
            } else {
                value.thumbUp += 1;
                dom.innerText = value.thumbUp.toString();
                dom.style.color = 'rgba(0,0,0,0.9)';
            }
            value.admirers.set(commentator, !isAdmired);
            // TODO 发送请求到后端，并更新数据库
            /**
             * @Tip 这里需要注意一点，点赞功能会存在恶意点击的情况，就是用户对当前评论频繁且快速点赞或取消点赞，如果每次点击我们都请求后端更新数据
             * 那么势必会影响服务器的性能，因此我们需要避免这一点，这里不做解释，大家有兴趣可以评论区留言探讨
             */
        }
    }
 
    showButton = function (arr, sign) {
 
        arr.forEach(function (value) {
 
            const parent = domById(value.id);
            const broEle = domById(value.id.substring(0, 6));
 
            const reply = domById(value.id.substring(0, 7));
 
            broEle.onmouseover = function (ev) {
                reply.style.display = 'inline-block';
            };
 
            thumbUp(value);
 
            reply.onclick = function (ev) {
                ev.stopPropagation();
                curArr = sign === 1 ? value.reply : arr;
                domById('cancel_reply').style.display = 'inline-block';
                const str = "回复 ".concat(value.commentator);
                fillPlaceholderOfCommentSpace(str)
            };
 
            const replyLen = value.reply.length;
            if (replyLen > 0) {
                handleReply(replyLen, value, broEle, parent);
            }
 
            broEle.onmouseleave = function (ev) {
                reply.style.display = 'none'
            };
        });
    };
 
    function handleReply(len, value, broEle, parent) {
        const checkReply = domById(value.id.substring(0, 8));
        checkReply.onclick = (ev) => {
            ev.stopPropagation();
            if(checkReply.innerText.trim().charAt(0) === '查'){
                ifHaveReply(parent, value.reply, broEle);
                checkReply.innerText = "收起回复";
            } else {
                toggleBackReplies(parent);
                checkReply.innerText = "查看回复("+ len +")";
            }
        };
    }
 
    toggleBackReplies = function (parentTag) {
        const nodes = parentTag.childNodes;
        parentTag.removeChild(nodes[nodes.length - 1]);
    };
 
    ifHaveReply = function (parentTag, arr, broEle) {
        const li = document.createElement("li");
        li.className = "reply_list";
        li.style.marginLeft = '42px';
        li.style.borderLeft = 'solid 5px lightgray';
 
        let htmlText = '<ul class="comment_ulist">';
 
        arr.forEach(function (value) {
            // 初始化当前角色是否给该条留言点赞了
            const commentator = domById('role_info_name').innerText;
            const isAdmirer = value.admirers.get(commentator);
            let color = 'rgba(0,0,0,0.9)';
            if (!isAdmirer) {
                color = 'rgba(0,0,0,0.4)';
            }
 
            htmlText +=
                `<li class="comment_line_box" id="${value.id.substring(0, 6)}">` +
                `   <div class="cmt_img" style="margin-left: 10px">` +
                `       <img alt="头像" class="avatar" src="img/avt.jpg" width="30px" style="border-radius: 100%">` +
                `   </div>` +
                `   <div class="cmt_info">` +
                `       <a class="commentator">${value.commentator}</a>` +
                `       <span>回复</span>` +
                `       <a class="author">${value.author}</a>` +
                `       <span class="time">${value.time}</span>` +
                `   </div>` +
                `   <div class="cmt_reply">` +
                `       <span id="${value.id.substring(0, 9)}">` +
                `           <a class="cmt_reply_btn" id="${value.id.substring(0, 7)}">回复</a>` +
                `       </span>` +
                `       <span>` +
                `           <a style="color:${color}" id="${value.id.substring(0, 10)}" class="thumb_up">${(value.thumbUp > 0 ? value.thumbUp : "")}</a>` +
                `       </span>` +
                `   </div>` +
                `   <div class="comment">${value.comment}</div>` +
                `</li>`;
        });
        htmlText += '</ul>';
        li.innerHTML = htmlText;
        parentTag.insertBefore(li, broEle.nextSibling);
        showButton(arr, 2)
    };
 
    fillPlaceholderOfCommentSpace = function (str) {
        document.getElementById('comment_space').placeholder = str
    };
 
    mainCLick = function (ev) {
        ev.stopPropagation();
        domById('emoji_board').style.display = 'none';
    };
 
    function len(id) {
        return domById(id).value.trim().length;
    }
 
    function domById(id) {
        return document.getElementById(id);
    }
