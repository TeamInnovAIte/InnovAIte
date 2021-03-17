import { Link } from 'gatsby'
import React from 'react'
import * as startStyles from '../styles/start.module.css'

export default function Home() {
    return (
        <div className = {startStyles.startPage}>
            <Link to='/front-page'>
                <div className={startStyles.startButton}>
                  Start Dashboard
                </div>
            </Link>
        </div>
    )
}
