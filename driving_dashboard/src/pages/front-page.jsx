import React from 'react'
import driveData from '../data/new_merged_data.json'
import * as mainStyles from '../styles/main.module.css'
import * as panelStyles from '../styles/panel.module.css'
import {FlexibleXYPlot, LineMarkSeries, VerticalBarSeries, XAxis, YAxis} from 'react-vis'
import logo from '../images/DriveAID.png'
import Popup from 'reactjs-popup'

document.body.style.overflow = "hidden"

const data = [
    {x: 0, y: 0},
    {x: 1, y: 0},
    {x: 2, y: 0},
    {x: 3, y: 0},
    {x: 4, y: 0},
    {x: 5, y: 0},
    {x: 6, y: 0}
];

const dayLabels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

function getAvg(score) {
    var total = 0;
    for(var i = 0; i < score.length; i++) {
        total += score[i];
    }
    return total/score.length;
}

function getAvgArr(arr, len) {
    var newArr = new Array(arr.length/len)
    for(var i = 0; i < newArr.length; i++) {
        newArr[i] = {x: i+1, y: getAvg(arr.slice(i*len, (i+1)*len))}
    }
    return newArr;
}

export default class Front extends React.Component {
    
    constructor(props) {
        super(props);
        this.state = {
            index: driveData.sessions.length - driveData.sessions.length%7,
            day: 0,
        };
    }

    changeWeek(direction) {
        var newIndex = this.state.index + direction*7;
        if(newIndex < 0)
            newIndex = 0;
        else if(newIndex >= driveData.sessions.length)
            newIndex -= 7;
        this.setState({
            index: newIndex
        });
    }

    render() {

        for(var i = 0; i < data.length; i++) {
            if(this.state.index+i >= 0 && this.state.index+i < driveData.sessions.length)
                data[i].y = getAvg(driveData.sessions[this.state.index+i].gesture_data.development[0].data)*100;
            else
                data[i].y = 0;
            
        }

        return (
            <div className = {mainStyles.pageStyle}>
                <div className = {mainStyles.mainTitle}>
                    <img src = {logo} alt = "Logo" className = {mainStyles.imageSize}/>
                    &nbsp;Driving Dashboard
                </div>
                <div className = {mainStyles.weekLabel}>
                    Week of {driveData.sessions[this.state.index].session_date}
                </div>
                <div className = {mainStyles.buttonRow}>
                    <button className = {mainStyles.backButton} onClick = {() => this.changeWeek(-1)}>
                        ❮
                    </button>
                    {dayLabels.map((option, index) => (
                        <Popup trigger = {
                            <button className = {mainStyles.buttonWrapper}>
                                {option}
                            </button>
                        }
                        modal
                        nested
                        >
                        {close => (
                            <div className = {panelStyles.popupWrapper}>
                                <div className = {panelStyles.closeButtonRow}>
                                    <button className = {panelStyles.closeButton} onClick = {close}>
                                            &times;
                                            Close
                                    </button>
                                </div>
                                <div className = {panelStyles.newButtonRow}>
                                    {driveData.sessions[index].gesture_data.development.map((category, iterate) => (
                                        <button className = {panelStyles.buttonWrapper}>
                                            {category.name}
                                        </button>
                                    ))}
                                </div>
                                <div className = {panelStyles.dataPlot}>
                                <FlexibleXYPlot yDomain={[0,100]}>
                                    <VerticalBarSeries data = {getAvgArr(driveData.sessions[index].gesture_data.development[0].data, 30)} color = 'blue'/>
                                    <LineMarkSeries data = {getAvgArr(driveData.sessions[index].gesture_data.development[0].data, 30)} color = 'red' style = {{fill: 'none', strokeWidth: 10}} curve = {'curveMonotoneX'}/>
                                    <YAxis/>
                                    <XAxis tickTotal = {driveData.sessions[index].gesture_data.development[0].data.length/30}/>
                                </FlexibleXYPlot>
                                </div>
                            </div>
                        )}
                        </Popup>
                    ))}
                    <button className = {mainStyles.forwardButton} onClick = {() => this.changeWeek(1)}>
                        ❯
                    </button>
                </div>
                <div className = {mainStyles.mainPlot}>
                    <FlexibleXYPlot yDomain={[0,100]}>
                        <VerticalBarSeries data = {data} color = 'blue'/>
                        <LineMarkSeries data = {data.slice(0, driveData.sessions.length-this.state.index)} color = 'red' style = {{fill: 'none', strokeWidth: 10}} curve = {'curveMonotoneX'}/>
                        <YAxis/>
                        <XAxis tickTotal = {7} tickFormat={d => {
                            if(this.state.index + d < driveData.sessions.length)
                                return driveData.sessions[this.state.index + d].session_date; 
                            return "-----";
                        }}/>
                    </FlexibleXYPlot>
                </div>
            </div>
        )
    }
}